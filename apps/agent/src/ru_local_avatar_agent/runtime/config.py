from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(os.getenv("RU_LOCAL_AVATAR_ROOT", Path.cwd())).resolve()


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    root: Path
    profile_path: Path
    profile: dict[str, Any]
    vllm_base_url: str
    llm_model: str
    voice_preset: str
    avatar_url: str
    livekit_host: str
    livekit_host_port: int
    livekit_ws_url: str
    livekit_public_ws_url: str
    voice_enabled: bool
    status_scan_sizes: bool
    min_free_vram_gb_after_load: float
    production: bool
    public_status: bool
    cors_origins: tuple[str, ...]
    trusted_hosts: tuple[str, ...]
    admin_api_token: str | None
    max_active_voice_sessions: int
    session_ttl_seconds: int
    conversation_audit_path: Path
    livekit_api_key: str | None
    livekit_api_secret: str | None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def load_runtime_config() -> RuntimeConfig:
    root = project_root()
    profile_value = os.getenv("RU_LOCAL_AVATAR_PROFILE", "profiles/rtx5090.yaml")
    profile_path = Path(profile_value)
    if not profile_path.is_absolute():
        profile_path = root / profile_path

    profile: dict[str, Any] = {}
    if profile_path.exists():
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}

    llm = profile.get("llm", {})
    tts = profile.get("tts", {})
    tts_voice = tts.get("voice", {})
    avatar = profile.get("avatar", {})
    runtime = profile.get("runtime", {})
    audit_path = Path(
        os.getenv(
            "RU_LOCAL_AVATAR_CONVERSATION_AUDIT_PATH",
            str(root / "data" / "conversation_audit.sqlite3"),
        )
    )
    if not audit_path.is_absolute():
        audit_path = root / audit_path

    return RuntimeConfig(
        root=root,
        profile_path=profile_path,
        profile=profile,
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:46111/v1").rstrip("/"),
        llm_model=os.getenv("QWEN_MODEL_ID", llm.get("default_hf_id", "Qwen/Qwen3-14B-FP8")),
        voice_preset=os.getenv("QWEN_TTS_VOICE_PRESET", tts_voice.get("preset", "qwen_default")),
        avatar_url=avatar.get(
            "browser_url",
            "/assets/avatars/default-female/AliciaSolid_vrm-0.51.vrm",
        ),
        livekit_host=os.getenv("LIVEKIT_HOST", "127.0.0.1"),
        livekit_host_port=int(os.getenv("LIVEKIT_HOST_PORT", "46280")),
        livekit_ws_url=os.getenv(
            "LIVEKIT_WS_URL",
            f"ws://{os.getenv('LIVEKIT_HOST', '127.0.0.1')}:{os.getenv('LIVEKIT_HOST_PORT', '46280')}",
        ),
        livekit_public_ws_url=os.getenv(
            "LIVEKIT_PUBLIC_WS_URL",
            f"ws://127.0.0.1:{os.getenv('LIVEKIT_HOST_PORT', '46280')}",
        ),
        voice_enabled=os.getenv("RU_LOCAL_AVATAR_VOICE", "auto") != "0",
        status_scan_sizes=os.getenv("RU_LOCAL_AVATAR_STATUS_SCAN_SIZES", "0") == "1",
        min_free_vram_gb_after_load=float(runtime.get("min_free_vram_gb_after_load", 2.0)),
        production=os.getenv("RU_LOCAL_AVATAR_ENV") == "production",
        public_status=_env_bool("RU_LOCAL_AVATAR_PUBLIC_STATUS"),
        cors_origins=_env_list(
            "RU_LOCAL_AVATAR_CORS_ORIGINS",
            (
                "http://127.0.0.1:46174",
                "http://localhost:46174",
            ),
        ),
        trusted_hosts=_env_list(
            "RU_LOCAL_AVATAR_TRUSTED_HOSTS",
            (
                "127.0.0.1",
                "localhost",
                "testserver",
            ),
        ),
        admin_api_token=os.getenv("RU_LOCAL_AVATAR_ADMIN_TOKEN") or None,
        max_active_voice_sessions=_env_int("RU_LOCAL_AVATAR_MAX_ACTIVE_VOICE_SESSIONS", 1),
        session_ttl_seconds=_env_int("RU_LOCAL_AVATAR_SESSION_TTL_SECONDS", 15 * 60),
        conversation_audit_path=audit_path,
        livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )


def resolve_profile_path(config: RuntimeConfig, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config.root / path
