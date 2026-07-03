"""Audio2Face-3D v3.0 ONNX inference + ARKit blendshape solve.

Replicates the streaming loop from NVIDIA's Audio2Face-3D-Training-Framework
(`infer_diffusion.py`, streaming_stateless ONNX mode):

- 16 kHz mono audio, window of 16000 samples, stride 8000,
- the network returns 60 geometry frames (60 fps) per window; frames
  [15:45) are kept, GRU latents are carried between windows,
- the stream is left-padded with 1 s of silence and the first 45 kept
  frames are dropped, so kept frame `i` corresponds to `i / 60` seconds,
- geometry is solved to 52 ARKit blendshape weights via regularized least
  squares over the frontal-mask vertices (per bs_skin_config).

TTS synthesizes faster than realtime, so clause audio is animated ahead of
playback; only the first window sits on the latency path (~10-30 ms on GPU).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

A2F_SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
STRIDE_SAMPLES = 8_000
PAD_SAMPLES = 16_000
LEFT_TRUNCATE = 15
BLOCK_FRAMES = 30
OUTPUT_FPS = 60
EMIT_FPS = 30
SKIN_VALUES = 72_006  # 24002 verts * 3


@dataclass(frozen=True, slots=True)
class FaceFrame:
    pts_ms: int
    values: dict[str, float]


class A2FEngine:
    def __init__(
        self,
        model_dir: Path,
        *,
        identity: str = "Claire",
        prefer_gpu: bool = True,
    ) -> None:
        # Import torch first so its bundled CUDA/cuDNN libraries are loaded
        # into the process before the ORT CUDA provider resolves them.
        import torch  # noqa: F401,I001
        import onnxruntime as ort

        self.model_dir = model_dir
        info = json.loads((model_dir / "network_info.json").read_text(encoding="utf-8"))
        params = info["params"]
        self.identities: list[str] = params["identities"]
        self.num_emotions: int = len(params["emotions"])
        self.num_diffusion_steps: int = params["num_diffusion_steps"]
        self.gru_layers: int = params["num_gru_layers"]
        self.gru_dim: int = params["gru_latent_dim"]
        self.total_values: int = (
            params["skin_size"] + params["tongue_size"] + params["jaw_size"] + params["eyes_size"]
        )
        if identity not in self.identities:
            raise ValueError(
                f"unknown A2F identity {identity!r}, expected one of {self.identities}"
            )
        self.identity = identity

        providers = ["CPUExecutionProvider"]
        self.provider = "CPUExecutionProvider"
        if prefer_gpu:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                raise RuntimeError(
                    "A2F CUDAExecutionProvider is not available; "
                    f"available providers: {available}"
                )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_dir / "network.onnx"), sess_options=options, providers=providers
        )
        self.provider = self.session.get_providers()[0]
        if prefer_gpu and self.provider != "CUDAExecutionProvider":
            raise RuntimeError(
                "A2F requested CUDAExecutionProvider but ONNX Runtime selected "
                f"{self.provider}"
            )
        logger.info("A2F ONNX session using %s", self.provider)

        self._identity_vec = np.zeros((1, len(self.identities)), dtype=np.float32)
        self._identity_vec[0, self.identities.index(identity)] = 1.0
        # Deterministic noise, generated once and reused for every window —
        # mirrors DETERMINISTIC_NOISE_PATH in the reference implementation.
        rng = np.random.default_rng(20260702)
        self._noise = rng.standard_normal(
            (1, self.num_diffusion_steps + 1, OUTPUT_FPS, self.total_values)
        ).astype(np.float32)

        self._solver = _BlendshapeSolver(model_dir, identity)

    @property
    def blendshape_names(self) -> list[str]:
        return self._solver.pose_names

    def initial_latents(self) -> np.ndarray:
        return np.zeros(
            (self.num_diffusion_steps, self.gru_layers, 1, self.gru_dim), dtype=np.float32
        )

    def infer_window(
        self,
        window: np.ndarray,
        latents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one 16000-sample window; returns (block geometry [30, V], new latents)."""
        from ru_local_avatar_agent.voice import metrics

        emotion = np.zeros((1, BLOCK_FRAMES, self.num_emotions), dtype=np.float32)
        started = time.monotonic()
        outputs = self.session.run(
            None,
            {
                "window": window.reshape(1, WINDOW_SAMPLES).astype(np.float32),
                "identity": self._identity_vec,
                "emotion": emotion,
                "input_latents": latents,
                "noise": self._noise,
            },
        )
        metrics.A2F_WINDOW_MS.observe((time.monotonic() - started) * 1000)
        prediction, new_latents = outputs[0], outputs[1]
        block = prediction[0, LEFT_TRUNCATE : LEFT_TRUNCATE + BLOCK_FRAMES]
        return block, new_latents.astype(np.float32)

    def solve_frames(self, geometry_frames: np.ndarray) -> list[dict[str, float]]:
        return self._solver.solve(geometry_frames[:, :SKIN_VALUES])


class A2FTurnStream:
    """Streams one response's audio through A2F, keeping window continuity."""

    def __init__(self, engine: A2FEngine) -> None:
        self.engine = engine
        self._latents = engine.initial_latents()
        self._buffer = np.zeros(PAD_SAMPLES, dtype=np.float32)  # left pad
        self._consumed = 0  # samples already used as window starts
        self._kept_frames = 0  # 60fps frames emitted so far (after global skip)
        self._skip_frames = PAD_SAMPLES * OUTPUT_FPS // A2F_SAMPLE_RATE - LEFT_TRUNCATE  # 45
        self._solver_smooth: dict[str, float] = {}

    def push_audio(self, pcm_16k: np.ndarray) -> list[FaceFrame]:
        self._buffer = np.concatenate([self._buffer, pcm_16k.astype(np.float32)])
        return self._process_ready()

    def flush(self) -> list[FaceFrame]:
        self._buffer = np.concatenate([self._buffer, np.zeros(PAD_SAMPLES, dtype=np.float32)])
        return self._process_ready()

    def _process_ready(self) -> list[FaceFrame]:
        frames: list[FaceFrame] = []
        while self._consumed + WINDOW_SAMPLES <= self._buffer.shape[0]:
            window = self._buffer[self._consumed : self._consumed + WINDOW_SAMPLES]
            block, self._latents = self.engine.infer_window(window, self._latents)
            self._consumed += STRIDE_SAMPLES
            frames.extend(self._emit_block(block))
        # Drop audio that can no longer be a window start to bound memory.
        if self._consumed > WINDOW_SAMPLES:
            drop = self._consumed - WINDOW_SAMPLES
            self._buffer = self._buffer[drop:]
            self._consumed -= drop
        return frames

    def _emit_block(self, block: np.ndarray) -> list[FaceFrame]:
        solved = self.engine.solve_frames(block)
        frames: list[FaceFrame] = []
        for values in solved:
            frame_index = self._kept_frames
            self._kept_frames += 1
            if frame_index < self._skip_frames:
                continue
            emitted_index = frame_index - self._skip_frames
            # 60 fps solve, 30 fps wire rate.
            if emitted_index % (OUTPUT_FPS // EMIT_FPS) != 0:
                continue
            values = self._smooth(values)
            pts_ms = emitted_index * 1000 // OUTPUT_FPS
            frames.append(FaceFrame(pts_ms=pts_ms, values=values))
        return frames

    def _smooth(self, values: dict[str, float]) -> dict[str, float]:
        alpha = 0.15
        if not self._solver_smooth:
            self._solver_smooth = dict(values)
            return values
        smoothed = {}
        for name, value in values.items():
            previous = self._solver_smooth.get(name, value)
            smoothed[name] = alpha * previous + (1.0 - alpha) * value
        self._solver_smooth = smoothed
        return smoothed


class _BlendshapeSolver:
    def __init__(self, model_dir: Path, identity: str) -> None:
        config = json.loads(
            (model_dir / f"bs_skin_config_{identity}.json").read_text(encoding="utf-8")
        )["blendshape_params"]
        data = np.load(model_dir / f"bs_skin_{identity}.npz")

        pose_names = [_decode_pose_name(name) for name in data["poseNames"]]
        if pose_names and pose_names[0] in {"neutral", "Neutral"}:
            pose_names = pose_names[1:]
        self.pose_names = pose_names

        mask = data["frontalMask"].astype(np.int64)
        neutral = data["neutral"][mask].reshape(-1).astype(np.float64)
        self._neutral = neutral

        active_flags = config["bsSolveActivePoses"]
        self._multipliers = np.asarray(config["bsWeightMultipliers"], dtype=np.float64)
        self._offsets = np.asarray(config["bsWeightOffsets"], dtype=np.float64)
        self._active = [
            i
            for i, name in enumerate(pose_names)
            if active_flags[i] == 1 and name in data.files
        ]
        deltas = np.stack(
            [
                (data[pose_names[i]][mask].reshape(-1).astype(np.float64) - neutral)
                for i in self._active
            ]
        )  # [K, M]
        self._deltas = deltas
        gram = deltas @ deltas.T
        l2 = float(config["strengthL2regularization"])
        ridge = l2 * np.trace(gram) / max(gram.shape[0], 1)
        self._gram_inv_a = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]), deltas
        )  # [K, M] so weights = gram_inv_a @ delta
        self._mask = mask
        self._is_delta_output: bool | None = None

        # Symmetric Left/Right pairs solved independently then softly tied.
        self._pairs: list[tuple[int, int]] = []
        name_to_active = {pose_names[i]: k for k, i in enumerate(self._active)}
        for name, k in name_to_active.items():
            if name.endswith("Left"):
                partner = name[: -len("Left")] + "Right"
                if partner in name_to_active:
                    self._pairs.append((k, name_to_active[partner]))

    def solve(self, skin_frames: np.ndarray) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        masked = skin_frames.reshape(skin_frames.shape[0], -1, 3)[:, self._mask, :].reshape(
            skin_frames.shape[0], -1
        ).astype(np.float64)
        if self._is_delta_output is None and masked.shape[0] > 0:
            # The exported network emits either absolute geometry or deltas
            # from neutral; detect once by comparing magnitudes.
            pred_norm = float(np.median(np.abs(masked)))
            neutral_norm = float(np.median(np.abs(self._neutral)))
            self._is_delta_output = pred_norm < 0.1 * neutral_norm
            logger.info(
                "A2F output mode detected: %s (pred %.4f vs neutral %.4f)",
                "delta" if self._is_delta_output else "absolute",
                pred_norm,
                neutral_norm,
            )
        for row in masked:
            delta = row if self._is_delta_output else row - self._neutral
            weights = self._gram_inv_a @ delta
            for left, right in self._pairs:
                tied = 0.5 * (weights[left] + weights[right])
                weights[left] = 0.5 * (weights[left] + tied)
                weights[right] = 0.5 * (weights[right] + tied)
            values: dict[str, float] = {}
            for k, pose_index in enumerate(self._active):
                name = self.pose_names[pose_index]
                value = weights[k] * self._multipliers[pose_index] + self._offsets[pose_index]
                if abs(value) < 0.02:
                    value = 0.0
                values[name] = float(np.clip(value, 0.0, 1.0))
            results.append(values)
        return results


def _decode_pose_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)
