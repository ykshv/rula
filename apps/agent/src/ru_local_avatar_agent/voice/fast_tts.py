"""Experimental Qwen3-TTS code-predictor fast path.

Qwen3-TTS calls a small nested ``GenerationMixin.generate()`` loop for every
audio-code group. That generic generation path is convenient but expensive for
short fixed-length decode. This module installs an explicit loop behind a
profile flag so we can benchmark the latency win without making a private API
dependency part of the default runtime.
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Any


class FastCodePredictorInstallError(RuntimeError):
    """Raised when the installed qwen-tts build is not compatible."""


def install_fast_code_predictor(tts_model: Any) -> bool:
    """Install a manual code-predictor decode loop on a loaded Qwen3-TTS model.

    The patch is intentionally fail-closed: if the qwen-tts internals do not
    match the expected structure, runtime startup must fail when the profile
    explicitly asks for this optimization.
    """

    code_predictor = _resolve_code_predictor(tts_model)
    if getattr(code_predictor, "_rula_fast_generate_installed", False):
        return True

    _validate_code_predictor(code_predictor)
    original_generate = getattr(code_predictor, "generate", None)
    if original_generate is None:
        raise FastCodePredictorInstallError("code_predictor.generate is missing")

    code_predictor._rula_original_generate = original_generate
    code_predictor.generate = types.MethodType(_fast_generate, code_predictor)
    code_predictor._rula_fast_generate_installed = True
    return True


def _resolve_code_predictor(tts_model: Any) -> Any:
    try:
        return tts_model.model.talker.code_predictor
    except AttributeError as exc:
        raise FastCodePredictorInstallError(
            "expected tts_model.model.talker.code_predictor"
        ) from exc


def _validate_code_predictor(code_predictor: Any) -> None:
    required = ("model", "lm_head", "small_to_mtp_projection")
    missing = [name for name in required if not hasattr(code_predictor, name)]
    if missing:
        raise FastCodePredictorInstallError(
            f"code_predictor is missing required attributes: {', '.join(missing)}"
        )

    get_input_embeddings = getattr(code_predictor.model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        raise FastCodePredictorInstallError("code_predictor.model.get_input_embeddings is missing")

    try:
        embed_layers = get_input_embeddings()
    except Exception as exc:  # pragma: no cover - depends on qwen-tts internals
        raise FastCodePredictorInstallError("failed to inspect input embeddings") from exc

    _require_non_empty_sequence(code_predictor.lm_head, "code_predictor.lm_head")
    _require_non_empty_sequence(embed_layers, "code_predictor input embeddings")

    try:
        if len(code_predictor.lm_head) != len(embed_layers):
            raise FastCodePredictorInstallError(
                "lm_head and input embedding layer counts differ "
                f"({len(code_predictor.lm_head)} != {len(embed_layers)})"
            )
    except TypeError:
        # Some torch containers may not expose len(); index errors are still
        # caught during benchmark/smoke before this flag is allowed in prod.
        pass


def _require_non_empty_sequence(value: Any, name: str) -> None:
    try:
        if len(value) < 1:
            raise FastCodePredictorInstallError(f"{name} is empty")
    except TypeError as exc:
        if not hasattr(value, "__getitem__"):
            raise FastCodePredictorInstallError(f"{name} is not indexable") from exc


def _normalize_token_ids(token_ids: int | Iterable[int] | None) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, int):
        return [token_ids]
    return [int(token_id) for token_id in token_ids]


def _filter_logits(
    logits: Any,
    *,
    top_k: int | None,
    top_p: float | None,
    suppress_tokens: int | Iterable[int] | None,
) -> Any:
    import torch

    if suppress_tokens is not None:
        ids = [
            token_id
            for token_id in _normalize_token_ids(suppress_tokens)
            if 0 <= token_id < logits.shape[-1]
        ]
        if ids:
            index = torch.tensor(ids, device=logits.device, dtype=torch.long)
            logits = logits.index_fill(-1, index, float("-inf"))

    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

    return logits


def _all_eos(token: Any, eos_token_ids: set[int]) -> bool:
    if not eos_token_ids:
        return False
    return all(int(value) in eos_token_ids for value in token.reshape(-1).tolist())


def _model_call(model: Any, **kwargs: Any) -> Any:
    filtered = {key: value for key, value in kwargs.items() if value is not None}
    return model(**filtered)


def _fast_generate(
    self: Any,
    *,
    inputs_embeds: Any | None = None,
    attention_mask: Any | None = None,
    max_new_tokens: int = 15,
    do_sample: bool = True,
    top_p: float = 1.0,
    top_k: int = 50,
    temperature: float = 0.9,
    eos_token_id: int | Iterable[int] | None = None,
    suppress_tokens: int | Iterable[int] | None = None,
    **_: Any,
) -> Any:
    import torch
    from transformers.cache_utils import DynamicCache

    if inputs_embeds is None:
        raise ValueError("fast Qwen3-TTS code predictor requires inputs_embeds")
    if inputs_embeds.ndim != 3:
        raise ValueError(f"expected inputs_embeds rank 3, got {inputs_embeds.ndim}")
    if inputs_embeds.shape[0] != 1:
        raise NotImplementedError("fast Qwen3-TTS code predictor supports batch_size=1 only")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    device = inputs_embeds.device
    past_key_values = DynamicCache()
    projected_embeds = self.small_to_mtp_projection(inputs_embeds)
    cache_position = torch.arange(projected_embeds.shape[1], device=device)
    output = _model_call(
        self.model,
        inputs_embeds=projected_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    hidden = output.last_hidden_state[:, -1]

    # Matches qwen-tts code-predictor forward(): during prefill it selects
    # lm_head[inputs_embeds_len - 2], then advances one codebook per token.
    generation_step = max(int(inputs_embeds.shape[1]) - 2, 0)
    embed_layers = self.model.get_input_embeddings()
    eos_token_ids = set(_normalize_token_ids(eos_token_id))
    tokens: list[Any] = []

    for _step in range(max_new_tokens):
        if generation_step >= len(self.lm_head):
            break

        logits = self.lm_head[generation_step](hidden)
        if do_sample:
            logits = logits / max(float(temperature), 1e-5)
            logits = _filter_logits(
                logits,
                top_k=top_k,
                top_p=top_p,
                suppress_tokens=suppress_tokens,
            )
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
        else:
            logits = _filter_logits(
                logits,
                top_k=top_k,
                top_p=top_p,
                suppress_tokens=suppress_tokens,
            )
            token = logits.argmax(dim=-1, keepdim=True)

        tokens.append(token)
        if _all_eos(token, eos_token_ids):
            break

        if len(tokens) >= max_new_tokens or generation_step + 1 >= len(self.lm_head):
            break

        input_embed = embed_layers[generation_step](token)
        projected_embed = self.small_to_mtp_projection(input_embed)
        cache_position = torch.tensor([projected_embeds.shape[1] + len(tokens) - 1], device=device)
        output = _model_call(
            self.model,
            inputs_embeds=projected_embed,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden = output.last_hidden_state[:, -1]
        generation_step += 1

    if not tokens:
        raise RuntimeError("fast Qwen3-TTS code predictor produced no tokens")

    return types.SimpleNamespace(sequences=torch.cat(tokens, dim=-1))
