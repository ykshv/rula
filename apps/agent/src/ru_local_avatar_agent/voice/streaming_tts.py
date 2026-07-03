"""Streaming Qwen3-TTS decode loop tuned for realtime use on WSL2/RTX 5090.

Why this exists
---------------
The stock ``qwen_tts`` path runs one nested HuggingFace ``generate()`` per
codec frame (16 codebooks -> 1 main forward + a 15-step sub-generate). In
eager mode under WSL2 that costs ~145 ms per 80 ms audio frame (RTF ~1.8):
the GPU is idle most of the time while Python launches thousands of tiny
kernels.

This module replaces it with:

- a hand-rolled decode loop over the talker + code predictor (no
  GenerationMixin per step),
- ``StaticCache`` + manually captured ``torch.cuda.CUDAGraph`` per step shape,
  so each decode step replays as a single graph launch. Manual capture is
  deliberate: inductor's cudagraph trees refuse graphs whose inputs are
  mutated (the KV cache) and its partitioner+autotuner corrupts interleaved
  graph outputs — shapes here are fully static, so we own the buffers,
- frame-level streaming: audio chunks are emitted every few codec frames via
  incremental codec decode, so first audio needs only ~6 frames instead of a
  full clause.

Sampling semantics match ``Qwen3TTSForConditionalGeneration.generate``:
temperature/top-k for both talkers, repetition penalty over the main
codebook history, suppress list for non-codec ids, min two frames before EOS.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FRAME_MS = 80  # 12.5 Hz codec
FIRST_CHUNK_FRAMES = 6  # first emit once ~480 ms of codes exist
CHUNK_FRAMES_TIGHT = 4  # cadence while the playback lead is small
CHUNK_FRAMES_RELAXED = 12  # cadence once we are comfortably ahead
COMFORT_LEAD_MS = 1_200
DECODE_GUARD_FRAMES = 3  # codec decoder lookahead guard for non-final decodes
DECODE_CONTEXT_FRAMES = 24  # left context for incremental window decode
CROSSFADE_SAMPLES = 192  # ~8 ms @24 kHz equal-power blend across chunk seams
MAX_FRAMES = 448  # ~36 s hard ceiling per clause
TALKER_CACHE_LEN = 1024
PREDICTOR_CACHE_LEN = 24


@dataclass(frozen=True, slots=True)
class TtsChunk:
    pcm: np.ndarray  # float32 mono at `sample_rate`
    sample_rate: int
    pts_ms: int  # offset of this chunk inside the clause
    is_final: bool


@dataclass(frozen=True, slots=True)
class TtsGenerationSettings:
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_do_sample: bool
    subtalker_top_k: int
    subtalker_top_p: float
    subtalker_temperature: float
    max_frames: int


def resolve_generation_settings(raw: dict | None) -> TtsGenerationSettings:
    config = raw or {}
    do_sample = _bool_config(config.get("do_sample", True))
    top_k = _non_negative_int(config.get("top_k", 50), "top_k")
    top_p = _probability(config.get("top_p", 1.0), "top_p")
    temperature = _positive_float(config.get("temperature", 0.9), "temperature")
    repetition_penalty = _positive_float(
        config.get("repetition_penalty", 1.05), "repetition_penalty"
    )
    subtalker_do_sample = _bool_config(
        config.get("subtalker_dosample", config.get("subtalker_do_sample", do_sample))
    )
    subtalker_top_k = _non_negative_int(config.get("subtalker_top_k", top_k), "subtalker_top_k")
    subtalker_top_p = _probability(config.get("subtalker_top_p", top_p), "subtalker_top_p")
    subtalker_temperature = _positive_float(
        config.get("subtalker_temperature", temperature), "subtalker_temperature"
    )
    max_frames = _positive_int(config.get("max_new_tokens", MAX_FRAMES), "max_new_tokens")
    return TtsGenerationSettings(
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        subtalker_do_sample=subtalker_do_sample,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
        max_frames=max_frames,
    )


def _bool_config(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"expected boolean config value, got {value!r}")


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _positive_float(value: object, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _probability(value: object, name: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return parsed


class _GraphedStep:
    """One manually captured CUDA graph for a fixed-shape decode step.

    The callable must be shape-stable and value-driven (positions/masks built
    from tensor ops, which holds for HF StaticCache attention). Inputs are
    copied into static buffers, the graph replays as one launch, outputs are
    cloned out of the graph-owned buffers.
    """

    def __init__(self, fn, example_args: tuple, *, warmup_iters: int = 3) -> None:
        import torch

        self._torch = torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_iters):
                fn(*example_args)
        torch.cuda.current_stream().wait_stream(stream)

        self._static_inputs = tuple(arg.clone() for arg in example_args)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            out = fn(*self._static_inputs)
        self._static_outputs = out if isinstance(out, tuple) else (out,)
        self._single = not isinstance(out, tuple)

    def __call__(self, *args):
        for buffer, arg in zip(self._static_inputs, args, strict=False):
            buffer.copy_(arg, non_blocking=True)
        self._graph.replay()
        outputs = tuple(o.clone() for o in self._static_outputs)
        return outputs[0] if self._single else outputs


class StreamingQwenTts:
    """Owns the loaded Qwen3-TTS model and streams PCM per clause.

    Not thread-safe by design: the voice runtime serializes calls through a
    single TTS executor thread.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        speaker: str = "Serena",
        language: str = "Russian",
        device: str = "cuda:0",
        compile_steps: bool = True,
        generation_kwargs: dict | None = None,
    ) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        self.device = device
        self.speaker = speaker
        self.language = language

        self._qwen = Qwen3TTSModel.from_pretrained(
            str(model_dir),
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self._model = self._qwen.model
        self._talker = self._model.talker
        self._cp = self._talker.code_predictor
        self._cfg = self._model.config.talker_config
        self._num_code_groups = int(self._cfg.num_code_groups)
        self._eos_id = int(self._cfg.codec_eos_token_id)

        gen_kwargs = self._qwen._merge_generate_kwargs()
        gen_kwargs.update(generation_kwargs or {})
        self.generation_settings = resolve_generation_settings(gen_kwargs)

        vocab = int(self._cfg.vocab_size)
        suppress = torch.zeros(vocab, dtype=torch.bool, device=device)
        suppress[vocab - 1024 :] = True
        suppress[self._eos_id] = False
        self._suppress_mask = suppress

        self._sample_rate: int | None = None
        self._lock = threading.Lock()

        self._talker_cache = None
        self._cp_cache = None
        self._talker_step_fn = None
        self._cp_step_fn = None
        self._compiled = False
        self._init_caches_and_compile(compile_steps)

    # ------------------------------------------------------------------ setup

    def _init_caches_and_compile(self, compile_steps: bool) -> None:
        import torch
        from transformers import StaticCache

        self._talker_cache = StaticCache(
            config=self._talker.config,
            max_batch_size=1,
            max_cache_len=TALKER_CACHE_LEN,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._cp_cache = StaticCache(
            config=self._cp.config,
            max_batch_size=1,
            max_cache_len=PREDICTOR_CACHE_LEN,
            device=self.device,
            dtype=torch.bfloat16,
        )

        def talker_step(inputs_embeds, cache_position, position_ids):
            out = self._talker.model(
                inputs_embeds=inputs_embeds,
                past_key_values=self._talker_cache,
                use_cache=True,
                cache_position=cache_position,
                position_ids=position_ids,
            )
            hidden = out.last_hidden_state
            return hidden, self._talker.codec_head(hidden[:, -1])

        def cp_step(inputs_embeds, cache_position):
            out = self._cp.model(
                inputs_embeds=inputs_embeds,
                past_key_values=self._cp_cache,
                use_cache=True,
                cache_position=cache_position,
            )
            return out.last_hidden_state[:, -1]

        self._talker_step_fn = talker_step
        self._cp_step_fn = cp_step
        if compile_steps:
            try:
                self._capture_graphs(talker_step, cp_step)
                self._compiled = True
            except Exception:
                logger.exception("CUDA graph capture failed; falling back to eager steps")
                self._talker_cache.reset()
                self._cp_cache.reset()
                self._talker_step_fn = talker_step
                self._cp_step_fn = cp_step

    def _capture_graphs(self, talker_step, cp_step) -> None:
        """Capture one graph per fixed step shape; caches are reset afterwards."""
        import torch

        device = self.device
        talker_hidden = int(self._talker.config.hidden_size)
        cp_hidden = int(self._cp.config.hidden_size)

        # no_grad, not inference_mode: StaticCache allocates its buffers
        # lazily on first update, and inference-mode-born tensors reject
        # later mutation (reset) outside the mode.
        with torch.no_grad():
            talker_graph = _GraphedStep(
                talker_step,
                (
                    torch.zeros(1, 1, talker_hidden, dtype=torch.bfloat16, device=device),
                    torch.tensor([64], device=device),
                    torch.tensor([64], device=device).view(1, 1, 1).expand(3, 1, 1).contiguous(),
                ),
            )
            cp_prefill_graph = _GraphedStep(
                cp_step,
                (
                    torch.zeros(1, 2, cp_hidden, dtype=torch.bfloat16, device=device),
                    torch.arange(2, device=device),
                ),
            )
            cp_decode_graph = _GraphedStep(
                cp_step,
                (
                    torch.zeros(1, 1, cp_hidden, dtype=torch.bfloat16, device=device),
                    torch.tensor([2], device=device),
                ),
            )
        # Warmup/capture wrote garbage into the KV caches.
        self._talker_cache.reset()
        self._cp_cache.reset()

        def graphed_cp_step(inputs_embeds, cache_position):
            if inputs_embeds.shape[1] == 1:
                return cp_decode_graph(inputs_embeds, cache_position)
            return cp_prefill_graph(inputs_embeds, cache_position)

        self._talker_step_fn = talker_graph
        self._cp_step_fn = graphed_cp_step

    def warmup(self) -> None:
        """Trigger compilation/capture with a tiny synthesis."""
        started = time.monotonic()
        chunks = list(self.stream("Привет! Это проверка голоса."))
        total = sum(len(c.pcm) for c in chunks)
        logger.info(
            "tts warmup done in %.1fs (%.2fs audio, compiled=%s)",
            time.monotonic() - started,
            total / (self._sample_rate or 24_000),
            self._compiled,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate or 24_000

    # ------------------------------------------------------------- prefill

    def _build_prefill(self, text: str):
        """Replicates the custom-voice non-streaming prefill embedding build."""
        torch = self._torch
        model = self._model
        talker = self._talker
        cfg = self._cfg
        config = model.config

        input_id = self._qwen._tokenize_texts([self._qwen._build_assistant_text(text)])[0]
        input_id = input_id.to(talker.device)

        spk_id = cfg.spk_id[self.speaker.lower()]
        speaker_embed = talker.get_input_embeddings()(
            torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype)
        )
        language_id = cfg.codec_language_id[self.language.lower()]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
                    device=talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        codec_prefill = torch.tensor(
            [[cfg.codec_think_id, cfg.codec_think_bos_id, language_id, cfg.codec_think_eos_id]],
            device=talker.device,
            dtype=input_id.dtype,
        )
        codec_embed_0 = talker.get_input_embeddings()(codec_prefill)
        codec_embed_1 = talker.get_input_embeddings()(
            torch.tensor(
                [[cfg.codec_pad_id, cfg.codec_bos_id]], device=talker.device, dtype=input_id.dtype
            )
        )
        codec_embed = torch.cat([codec_embed_0, speaker_embed.view(1, 1, -1), codec_embed_1], dim=1)

        role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
        body = torch.cat(
            (tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos_embed), dim=1
        ) + codec_embed[:, :-1]
        prefill = torch.cat((role_embed, body), dim=1)

        # Non-streaming text mode: the whole text goes into the prefill.
        text_embeds = talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5]))
        codec_pads = talker.get_input_embeddings()(
            torch.full(
                (1, text_embeds.shape[1] + 1),
                cfg.codec_pad_id,
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
        prefill = torch.cat(
            [
                prefill,
                torch.cat((text_embeds, tts_eos_embed), dim=1) + codec_pads,
                tts_pad_embed
                + talker.get_input_embeddings()(
                    torch.tensor([[cfg.codec_bos_id]], device=talker.device, dtype=input_id.dtype)
                ),
            ],
            dim=1,
        )
        return prefill, tts_pad_embed

    # -------------------------------------------------------------- sampling

    def _sample(
        self,
        logits,
        *,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        history=None,
        ban_eos=False,
    ):
        torch = self._torch
        logits = logits.float()
        if history is not None:
            logits = logits.clone()
            score = logits[0, history]
            logits[0, history] = torch.where(
                score > 0,
                score / self.generation_settings.repetition_penalty,
                score * self.generation_settings.repetition_penalty,
            )
            logits[0] = logits[0].masked_fill(self._suppress_mask, float("-inf"))
            if ban_eos:
                logits[0, self._eos_id] = float("-inf")
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / max(temperature, 1e-5)
        if 0 < top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1,
                sorted_indices,
                sorted_logits,
            )
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    # ------------------------------------------------------------ subtalker

    def _predict_codes(self, past_hidden, last_id_hidden):
        """15 residual codebooks for one frame; returns (codes, summed embeds)."""
        torch = self._torch
        self._cp_cache.reset()
        prefill = self._cp.small_to_mtp_projection(
            torch.cat((past_hidden, last_id_hidden), dim=1)
        )
        # The 2-token prefill goes through the compiled step fn too; a second
        # CUDA graph is captured for the [1, 2, H] shape during warmup.
        hidden = self._cp_step_fn(prefill, torch.arange(2, device=self.device))

        embed_layers = self._cp.model.get_input_embeddings()
        codes = []
        embed_sum = last_id_hidden.squeeze(1).clone()
        steps = self._num_code_groups - 1
        for step in range(steps):
            token = self._sample(
                self._cp.lm_head[step](hidden),
                do_sample=self.generation_settings.subtalker_do_sample,
                top_k=self.generation_settings.subtalker_top_k,
                top_p=self.generation_settings.subtalker_top_p,
                temperature=self.generation_settings.subtalker_temperature,
            )
            codes.append(token)
            embed = embed_layers[step](token)
            embed_sum += embed.squeeze(1)
            if step == steps - 1:
                break
            hidden = self._cp_step_fn(
                self._cp.small_to_mtp_projection(embed),
                torch.tensor([2 + step], device=self.device),
            )
        return torch.cat(codes, dim=-1), embed_sum.unsqueeze(1)

    # ------------------------------------------------------------- streaming

    def stream(
        self,
        text: str,
        *,
        max_frames: int | None = None,
        should_stop=None,
    ) -> Iterator[TtsChunk]:
        """Synthesize `text`, yielding PCM chunks as codec frames become ready.

        `should_stop` is polled at every codec frame (~30 ms cadence), so a
        barge-in cancels synthesis long before the clause completes.
        """
        text = text.strip()
        if not text:
            return
        torch = self._torch
        # no_grad (not inference_mode): the generator suspends inside this
        # context, and inference_mode's thread-local flag plus its immutable
        # "inference tensors" interact badly with the static KV caches. The
        # generator body runs on the dedicated TTS thread, so the flag never
        # leaks into unrelated code.
        with self._lock, torch.no_grad():
            yield from self._stream_locked(
                text,
                max_frames or self.generation_settings.max_frames,
                should_stop,
            )

    def _stream_locked(self, text: str, max_frames: int, should_stop=None) -> Iterator[TtsChunk]:
        torch = self._torch
        talker = self._talker

        prefill, tts_pad_embed = self._build_prefill(text)
        self._talker_cache.reset()
        length = prefill.shape[1]
        positions = torch.arange(length, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        out = talker.model(
            inputs_embeds=prefill,
            past_key_values=self._talker_cache,
            use_cache=True,
            cache_position=torch.arange(length, device=self.device),
            position_ids=positions,
        )
        past_hidden = out.last_hidden_state[:, -1:]
        logits = talker.codec_head(out.last_hidden_state[:, -1])

        history: list[int] = []
        frames: list = []  # [1, 16] tensors
        emitted_frames = 0
        first_emit_wall: float | None = None
        seam_reserve: np.ndarray | None = None  # held-back tail for crossfade

        history_tensor = torch.empty(0, dtype=torch.long, device=self.device)
        for frame_index in range(max_frames):
            if should_stop is not None and should_stop():
                return
            token = self._sample(
                logits,
                do_sample=self.generation_settings.do_sample,
                top_k=self.generation_settings.top_k,
                top_p=self.generation_settings.top_p,
                temperature=self.generation_settings.temperature,
                history=history_tensor,
                ban_eos=frame_index < 2,
            )
            token_id = int(token.item())
            if token_id == self._eos_id:
                break
            history.append(token_id)
            history_tensor = torch.tensor(history, dtype=torch.long, device=self.device)

            last_id_hidden = talker.get_input_embeddings()(token)
            codes, embed_sum = self._predict_codes(past_hidden, last_id_hidden)
            frames.append(torch.cat((token, codes), dim=-1))

            step_embeds = embed_sum + tts_pad_embed
            cache_position = torch.tensor([length + frame_index], device=self.device)
            hidden, logits = self._talker_step_fn(
                step_embeds,
                cache_position,
                cache_position.view(1, 1, 1).expand(3, 1, 1),
            )
            past_hidden = hidden[:, -1:]

            emittable = len(frames) - DECODE_GUARD_FRAMES - emitted_frames
            if emitted_frames == 0:
                boundary = len(frames) >= FIRST_CHUNK_FRAMES
            else:
                lead_ms = emitted_frames * FRAME_MS - (
                    (time.monotonic() - first_emit_wall) * 1000
                )
                cadence = (
                    CHUNK_FRAMES_TIGHT if lead_ms < COMFORT_LEAD_MS else CHUNK_FRAMES_RELAXED
                )
                boundary = emittable >= cadence
            if boundary and emittable > 0:
                region, prev_tail = self._decode_window(frames, emitted_frames, emittable)
                pcm, seam_reserve = self._splice(seam_reserve, prev_tail, region, final=False)
                if first_emit_wall is None:
                    first_emit_wall = time.monotonic()
                yield TtsChunk(
                    pcm=pcm,
                    sample_rate=self.sample_rate,
                    pts_ms=emitted_frames * FRAME_MS,
                    is_final=False,
                )
                emitted_frames += emittable

        remaining = len(frames) - emitted_frames
        if remaining > 0:
            region, prev_tail = self._decode_window(frames, emitted_frames, remaining)
            pcm, _ = self._splice(seam_reserve, prev_tail, region, final=True)
        elif seam_reserve is not None:
            pcm = seam_reserve
        else:
            pcm = np.zeros(0, dtype=np.float32)
        yield TtsChunk(
            pcm=pcm,
            sample_rate=self.sample_rate,
            pts_ms=emitted_frames * FRAME_MS,
            is_final=True,
        )

    @staticmethod
    def _splice(
        reserve: np.ndarray | None,
        prev_tail: np.ndarray,
        region: np.ndarray,
        *,
        final: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Join chunks click-free.

        The last CROSSFADE_SAMPLES of every emit are held back; the next
        decode re-synthesizes those samples with fresher right context
        (`prev_tail`) and the two versions are equal-power blended.
        """
        parts: list[np.ndarray] = []
        if reserve is not None:
            n = min(len(reserve), len(prev_tail))
            if n > 0:
                fade_in = np.sin(np.linspace(0.0, np.pi / 2, n, dtype=np.float32)) ** 2
                blended = reserve[-n:] * (1.0 - fade_in) + prev_tail[-n:] * fade_in
                parts.append(reserve[: len(reserve) - n])
                parts.append(blended.astype(np.float32))
            else:
                parts.append(reserve)
        if final or len(region) <= CROSSFADE_SAMPLES:
            parts.append(region)
            return np.concatenate(parts) if parts else region, None
        parts.append(region[:-CROSSFADE_SAMPLES])
        return np.concatenate(parts), region[-CROSSFADE_SAMPLES:].copy()

    def _decode_window(
        self, frames: list, emitted_frames: int, new_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode `new_frames` starting at `emitted_frames` with rolling left context.

        The 12 Hz decoder maps frames to samples exactly (1920 samples/frame at
        24 kHz), so chunk boundaries are frame-aligned; the context window keeps
        the decoder's receptive field warm across chunk seams.

        Returns (new_region, prev_tail) where prev_tail is this decode's
        re-synthesis of the samples just before the seam, used for crossfading.
        """
        torch = self._torch
        context = min(emitted_frames, DECODE_CONTEXT_FRAMES)
        start = emitted_frames - context
        end = emitted_frames + new_frames
        codes = torch.cat(frames[start:end], dim=0)  # [context + new, 16]
        wavs, sample_rate = self._model.speech_tokenizer.decode([{"audio_codes": codes}])
        self._sample_rate = int(sample_rate)
        pcm = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        samples_per_frame = int(round(self._sample_rate * FRAME_MS / 1000))
        seam = context * samples_per_frame
        region = pcm[seam : seam + new_frames * samples_per_frame]
        prev_tail = pcm[max(0, seam - CROSSFADE_SAMPLES) : seam]
        return region, prev_tail
