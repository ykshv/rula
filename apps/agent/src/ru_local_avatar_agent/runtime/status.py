from __future__ import annotations

import json
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ru_local_avatar_agent.runtime.config import RuntimeConfig, resolve_profile_path


@dataclass(frozen=True, slots=True)
class Check:
    ok: bool
    detail: str = ""


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        relative_parts = set(item.relative_to(path).parts)
        if ".cache" in relative_parts or item.suffix in {".incomplete", ".metadata", ".lock"}:
            continue
        total += item.stat().st_size
    return total


def _hf_local_dir(models_root: Path, repo_id: str, fallback: str) -> Path:
    return models_root / repo_id.replace("/", "__") if repo_id else models_root / fallback


def artifact_checks(config: RuntimeConfig) -> dict[str, dict[str, Any]]:
    profile = config.profile
    models_root = config.root / "models" / "hf"
    tts_model_id = profile.get("tts", {}).get("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    llm_model_id = profile.get("llm", {}).get("default_hf_id", "Qwen/Qwen3-14B-FP8")
    avatar_manifest_path = resolve_profile_path(
        config,
        profile.get("avatar", {}).get(
            "manifest",
            "models/manifests/default_avatar.alicia_solid_vrm_0_51.json",
        ),
    )

    expected = {
        "stt": models_root / "ai-sage__GigaAM-v3",
        "tts": _hf_local_dir(models_root, tts_model_id, "Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "a2f": models_root / "nvidia__Audio2Face-3D-v3.0",
        "llm": _hf_local_dir(models_root, llm_model_id, "Qwen__Qwen3-14B-FP8"),
    }

    result: dict[str, dict[str, Any]] = {}
    scan_sizes = getattr(config, "status_scan_sizes", False)
    for name, path in expected.items():
        result[name] = {
            "ok": path.exists() and any(path.iterdir()),
            "path": str(path),
            "size_gb": round(_dir_size_bytes(path) / 1024**3, 3)
            if scan_sizes and path.exists()
            else None,
        }

    avatar_ok = False
    avatar_detail = "manifest_missing"
    avatar_path = None
    if avatar_manifest_path.exists():
        manifest = json.loads(avatar_manifest_path.read_text(encoding="utf-8"))
        avatar_path = resolve_profile_path(config, manifest["artifact"]["web_public_path"])
        avatar_ok = avatar_path.exists() and avatar_path.stat().st_size == manifest["artifact"]["size_bytes"]
        avatar_detail = "ok" if avatar_ok else "missing_or_size_mismatch"

    result["avatar"] = {
        "ok": avatar_ok,
        "path": str(avatar_path) if avatar_path else "",
        "manifest": str(avatar_manifest_path),
        "detail": avatar_detail,
    }
    return result


def tcp_check(host: str, port: int, timeout_seconds: float = 0.4) -> Check:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return Check(ok=True, detail="tcp_open")
    except OSError as exc:
        return Check(ok=False, detail=str(exc))


def http_json(url: str, timeout_seconds: float = 0.8) -> tuple[bool, Any, str]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ru-local-avatar-agent/0.1"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return True, json.loads(raw) if raw else {}, ""
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return False, None, str(exc)


def gpu_check(config: RuntimeConfig) -> dict[str, Any]:
    min_free_gb = float(getattr(config, "min_free_vram_gb_after_load", 2.0))
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.5,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "detail": f"nvidia-smi unavailable: {exc}",
            "min_free_gb": min_free_gb,
        }
    if not output:
        return {"ok": False, "detail": "nvidia-smi returned no GPUs", "min_free_gb": min_free_gb}

    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) != 4:
        return {
            "ok": False,
            "detail": f"unexpected nvidia-smi output: {output.splitlines()[0]}",
            "min_free_gb": min_free_gb,
        }
    name, total_mb, used_mb, free_mb = parts
    free_gb = round(int(free_mb) / 1024, 3)
    used_gb = round(int(used_mb) / 1024, 3)
    total_gb = round(int(total_mb) / 1024, 3)
    return {
        "ok": free_gb >= min_free_gb,
        "detail": "headroom_ok" if free_gb >= min_free_gb else "insufficient_vram_headroom",
        "name": name,
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "min_free_gb": min_free_gb,
    }


def service_checks(config: RuntimeConfig, voice_runtime: Any | None = None) -> dict[str, dict[str, Any]]:
    livekit_credentials = bool(config.livekit_api_key and config.livekit_api_secret)
    livekit_tcp = tcp_check(config.livekit_host, config.livekit_host_port)
    vllm_ok, models_payload, vllm_detail = http_json(f"{config.vllm_base_url}/models")
    if voice_runtime is not None:
        voice_pipeline = voice_runtime.snapshot()
    else:
        voice_pipeline = {
            "ok": False,
            "detail": "voice runtime disabled in this process (dev shell); run the agent container",
        }
    # The profile gate is `min_free_vram_gb_after_load`: a one-time
    # post-load invariant. A live nvidia-smi comparison flaps under WSL2
    # because the number includes Windows desktop apps (the user's own
    # browser rendering the avatar takes VRAM and would lock voice out).
    cached_gpu = getattr(voice_runtime, "gpu_headroom_after_load", None)
    if cached_gpu is not None:
        gpu = dict(cached_gpu)
        gpu["measured"] = "after_load"
        live = gpu_check(config)
        gpu["live_free_gb"] = live.get("free_gb")
    else:
        gpu = gpu_check(config)
        gpu["measured"] = "live"
    return {
        "livekit_credentials": {
            "ok": livekit_credentials,
            "detail": "configured" if livekit_credentials else "missing",
        },
        "livekit_server": {
            "ok": livekit_tcp.ok,
            "host": config.livekit_host,
            "port": config.livekit_host_port,
            "detail": livekit_tcp.detail,
        },
        "vllm": {
            "ok": vllm_ok,
            "base_url": config.vllm_base_url,
            "detail": "ok" if vllm_ok else vllm_detail,
            "models": models_payload.get("data", []) if isinstance(models_payload, dict) else [],
        },
        "gpu": gpu,
        "voice_pipeline": voice_pipeline,
    }


def runtime_status(config: RuntimeConfig, voice_runtime: Any | None = None) -> dict[str, Any]:
    artifacts = artifact_checks(config)
    services = service_checks(config, voice_runtime)
    all_artifacts_ready = all(item["ok"] for item in artifacts.values())
    text_chat_ready = artifacts["llm"]["ok"] and services["vllm"]["ok"]
    voice_ready = (
        all_artifacts_ready
        and services["vllm"]["ok"]
        and services["gpu"]["ok"]
        and services["livekit_credentials"]["ok"]
        and services["livekit_server"]["ok"]
        and services["voice_pipeline"]["ok"]
    )
    return {
        "mode": "production" if config.production else "development",
        "profile_path": str(config.profile_path),
        "voice_preset": config.voice_preset,
        "avatar_url": config.avatar_url,
        "artifacts": artifacts,
        "services": services,
        "ready": {
            "text_chat": text_chat_ready,
            "voice_avatar": voice_ready,
            "all_artifacts": all_artifacts_ready,
        },
    }
