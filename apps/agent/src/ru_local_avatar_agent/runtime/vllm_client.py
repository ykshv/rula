from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from ru_local_avatar_agent.domain.contracts import DialogueModel, TokenDelta


class DialogueModelUnavailable(RuntimeError):
    pass


_client_lock = None
_shared_clients: dict[str, httpx.AsyncClient] = {}


def _shared_client(base_url: str, timeout: httpx.Timeout) -> httpx.AsyncClient:
    """Process-wide connection pool per endpoint.

    Creating a client per request added a TCP connect to every turn's
    time-to-first-token; the realtime loop cannot afford it.
    """
    client = _shared_clients.get(base_url)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout)
        _shared_clients[base_url] = client
    return client


@dataclass(frozen=True, slots=True)
class VllmDialogueModel(DialogueModel):
    """True token-streaming client for the vLLM OpenAI-compatible endpoint.

    Cancellation contract: aborting the consuming task closes the HTTP
    stream, which makes vLLM abort the request server-side. This is how
    `generation_id` bumps translate into freed GPU time.
    """

    base_url: str
    model: str
    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 5.0
    max_tokens: int = 384
    temperature: float = 0.6
    # Penalizes tokens already present in the context (including the
    # avatar's own previous replies), which breaks the "same opener every
    # turn" mode-collapse of short assistant answers.
    presence_penalty: float = 0.0

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[TokenDelta]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout_seconds)
        got_content = False
        client = _shared_client(self.base_url, timeout)
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"User-Agent": "ru-local-avatar-agent/0.1"},
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")[:400]
                    raise DialogueModelUnavailable(
                        f"vLLM returned HTTP {response.status_code}: {detail}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices") or []
                        delta = choices[0].get("delta") or {} if choices else {}
                    except (json.JSONDecodeError, AttributeError, IndexError) as exc:
                        raise DialogueModelUnavailable(
                            f"vLLM returned a malformed stream chunk: {data[:200]}"
                        ) from exc
                    content = delta.get("content")
                    if content:
                        got_content = True
                        yield TokenDelta(text=content)
        except httpx.HTTPError as exc:
            raise DialogueModelUnavailable(str(exc)) from exc

        if not got_content:
            raise DialogueModelUnavailable("vLLM stream produced no assistant tokens")
        yield TokenDelta(text="", is_final=True)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        chunks = [token.text async for token in self.stream(messages)]
        return "".join(chunks).strip()
