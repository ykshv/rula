from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ru_local_avatar_agent.domain.contracts import TranscriptDelta
from ru_local_avatar_agent.domain.events import EventKind, StreamEnvelope
from ru_local_avatar_agent.domain.session import InvalidTransition, SessionStateMachine
from ru_local_avatar_agent.metrics.acceptance import AcceptanceThresholds
from ru_local_avatar_agent.runtime.config import RuntimeConfig, load_runtime_config
from ru_local_avatar_agent.runtime.status import runtime_status
from ru_local_avatar_agent.runtime.vllm_client import DialogueModelUnavailable, VllmDialogueModel

logger = logging.getLogger(__name__)


class SessionResponse(BaseModel):
    session_id: str
    room_name: str
    livekit_token: str
    livekit_ws_url: str = ""
    livekit_public_ws_url: str = ""
    avatar_url: str = ""
    voice_enabled: bool = False
    voice_disabled_reason: str = "voice pipeline is not implemented yet"
    data_channel_required: bool = False
    hot_path_websocket_enabled: bool = False
    barge_in_enabled: bool = True


class SessionRequest(BaseModel):
    barge_in_enabled: bool = True


class CancelResponse(BaseModel):
    session_id: str
    generation_id: int
    state: str


class ConversationEventsResponse(BaseModel):
    session_id: str
    events: list[dict[str, Any]]


class TurnTraceResponse(BaseModel):
    session_id: str
    turns: list[dict[str, Any]]


class RuntimeStatusResponse(BaseModel):
    mode: str
    profile_path: str
    voice_preset: str
    avatar_url: str
    artifacts: dict[str, dict[str, Any]]
    services: dict[str, dict[str, Any]]
    ready: dict[str, bool]


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
    detail: str = Field(default="")


class TextChatRequest(BaseModel):
    session_id: str | None = None
    text: str = Field(min_length=1, max_length=4000)


class TextChatResponse(BaseModel):
    session_id: str
    turn_id: int
    generation_id: int
    branch_state: str
    assistant_text: str
    events: list[dict[str, Any]]


@dataclass(slots=True)
class StoredSession:
    session: SessionStateMachine
    created_at: float
    last_seen_at: float
    voice: bool


class SessionStore:
    def __init__(self, *, max_voice_sessions: int, ttl_seconds: int) -> None:
        self._sessions: dict[str, StoredSession] = {}
        self._max_voice_sessions = max(0, max_voice_sessions)
        self._ttl_seconds = max(60, ttl_seconds)

    def create(
        self,
        *,
        voice: bool = False,
        enforce_voice_limit: bool = False,
    ) -> SessionStateMachine:
        self._cleanup()
        if (
            voice
            and enforce_voice_limit
            and self._max_voice_sessions > 0
            and self._active_voice_count() >= self._max_voice_sessions
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="voice session limit reached",
            )
        session_id = str(uuid4())
        session = SessionStateMachine(session_id)
        now = time.monotonic()
        self._sessions[session_id] = StoredSession(
            session=session,
            created_at=now,
            last_seen_at=now,
            voice=voice,
        )
        return session

    def get(self, session_id: str) -> SessionStateMachine:
        self._cleanup()
        try:
            stored = self._sessions[session_id]
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            ) from exc
        stored.last_seen_at = time.monotonic()
        return stored.session

    def release_voice(self, session_id: str) -> None:
        self._cleanup()
        stored = self._sessions.get(session_id)
        if stored is not None:
            stored.voice = False
            stored.last_seen_at = time.monotonic()

    def _active_voice_count(self) -> int:
        return sum(1 for stored in self._sessions.values() if stored.voice)

    def _cleanup(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, stored in self._sessions.items()
            if now - stored.last_seen_at > self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)


def _public_runtime_status(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    sanitized["profile_path"] = ""
    sanitized["artifacts"] = {
        name: {
            key: value
            for key, value in artifact.items()
            if key not in {"path", "manifest"}
        }
        for name, artifact in payload.get("artifacts", {}).items()
    }
    sanitized["services"] = {
        name: {
            key: value
            for key, value in service.items()
            if key not in {"base_url", "models"}
        }
        for name, service in payload.get("services", {}).items()
    }
    return sanitized


def _runtime_status_payload(config: RuntimeConfig, voice_runtime: Any | None) -> dict[str, Any]:
    payload = runtime_status(config, voice_runtime)
    return _public_runtime_status(payload) if config.public_status else payload


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _require_admin_access(config: RuntimeConfig, request: Request) -> None:
    if not config.production:
        return
    supplied_token = _extract_bearer_token(request.headers.get("authorization"))
    expected_token = config.admin_api_token
    if not expected_token or not supplied_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    if not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")


def _try_create_voice_runtime(config: RuntimeConfig):
    """Voice runtime requires the GPU container deps; fail closed elsewhere."""
    if not config.voice_enabled:
        return None
    try:
        import torch  # noqa: F401

        from ru_local_avatar_agent.voice.runtime import VoiceRuntime
    except ImportError:
        return None
    return VoiceRuntime(config)


async def _load_voice_after_vllm_ready(config: RuntimeConfig, voice_runtime: Any) -> None:
    while True:
        status_payload = runtime_status(config, voice_runtime)
        if status_payload["services"]["vllm"]["ok"]:
            logger.info("vLLM is ready; loading in-process voice engines")
            voice_runtime.start_loading()
            return
        await asyncio.sleep(2.0)


def create_app() -> FastAPI:
    config: RuntimeConfig = load_runtime_config()
    app = FastAPI(
        title="Ru Local Avatar Agent",
        version="0.1.0",
        docs_url=None if config.production else "/docs",
        redoc_url=None if config.production else "/redoc",
        openapi_url=None if config.production else "/openapi.json",
    )
    if config.production:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.trusted_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    store = SessionStore(
        max_voice_sessions=config.max_active_voice_sessions,
        ttl_seconds=config.session_ttl_seconds,
    )
    thresholds = AcceptanceThresholds()
    voice_runtime = _try_create_voice_runtime(config)

    @app.on_event("startup")
    async def _startup() -> None:
        if voice_runtime is not None:
            app.state.voice_load_task = asyncio.create_task(
                _load_voice_after_vllm_ready(config, voice_runtime)
            )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        load_task = getattr(app.state, "voice_load_task", None)
        if load_task is not None:
            load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await load_task
        if voice_runtime is not None:
            await voice_runtime.shutdown()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        status_payload = _runtime_status_payload(config, voice_runtime)
        has_profile = config.profile_path.exists()
        ready_value = bool(status_payload["ready"]["voice_avatar"])
        return ReadyResponse(
            ready=ready_value,
            checks={
                "profile_configured": has_profile,
                "models_downloaded": bool(status_payload["ready"]["all_artifacts"]),
                "vllm_ready": bool(status_payload["services"]["vllm"]["ok"]),
                "gpu_headroom": bool(status_payload["services"]["gpu"]["ok"]),
                "livekit_credentials": bool(
                    status_payload["services"]["livekit_credentials"]["ok"]
                ),
                "livekit_server": bool(status_payload["services"]["livekit_server"]["ok"]),
                "voice_pipeline": bool(status_payload["services"]["voice_pipeline"]["ok"]),
                "text_chat_ready": bool(status_payload["ready"]["text_chat"]),
                "voice_avatar_ready": bool(status_payload["ready"]["voice_avatar"]),
            },
            detail=(
                ""
                if ready_value
                else (
                    "product readiness requires browser media, LiveKit agent loop, "
                    "STT, TTS, and A2F hot path"
                )
            ),
        )

    @app.get("/api/runtime/status", response_model=RuntimeStatusResponse)
    async def get_runtime_status() -> RuntimeStatusResponse:
        return RuntimeStatusResponse(**_runtime_status_payload(config, voice_runtime))

    @app.get("/metrics")
    async def metrics() -> Response:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            payload = generate_latest()
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            return Response(
                content="# prometheus_client is unavailable in this process\n",
                media_type="text/plain",
            )

    @app.post("/api/sessions", response_model=SessionResponse)
    async def create_session(request: SessionRequest | None = None) -> SessionResponse:
        if config.production and not (config.livekit_api_key and config.livekit_api_secret):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="production mode requires LiveKit credentials",
        )
        status_payload = _runtime_status_payload(config, voice_runtime)
        voice_enabled = bool(status_payload["ready"]["voice_avatar"])
        barge_in_enabled = request.barge_in_enabled if request is not None else True
        session = store.create(voice=voice_enabled, enforce_voice_limit=config.production)
        room_name = f"ru-local-avatar-{session.session_id}"
        disabled_reason = (
            "" if voice_enabled else status_payload["services"]["voice_pipeline"]["detail"]
        )
        livekit_token = ""
        livekit_ws_url = ""
        if voice_enabled and voice_runtime is not None:
            livekit_token = voice_runtime.mint_token(
                identity="user", name="User", room_name=room_name
            )
            livekit_ws_url = config.livekit_public_ws_url
            started = voice_runtime.start_worker(
                session,
                room_name,
                on_stop=store.release_voice,
                barge_in_enabled=barge_in_enabled,
            )
            if not started:
                store.release_voice(session.session_id)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="voice worker already exists for this session",
                )
        return SessionResponse(
            session_id=session.session_id,
            room_name=room_name,
            livekit_token=livekit_token,
            livekit_ws_url=livekit_ws_url,
            livekit_public_ws_url=config.livekit_public_ws_url if voice_enabled else "",
            avatar_url=config.avatar_url,
            voice_enabled=voice_enabled,
            voice_disabled_reason=disabled_reason,
            data_channel_required=voice_enabled,
            barge_in_enabled=barge_in_enabled,
        )

    @app.post("/api/chat/text", response_model=TextChatResponse)
    async def chat_text(request: TextChatRequest) -> TextChatResponse:
        status_payload = _runtime_status_payload(config, voice_runtime)
        if not status_payload["ready"]["text_chat"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "text chat requires downloaded LLM and running vLLM",
                    "runtime": status_payload,
                },
            )

        session = store.get(request.session_id) if request.session_id else store.create()
        if session.branch_state.value != "listening":
            session.interrupt("new_text_turn")

        try:
            ctx = session.commit_eot("text_input")
        except InvalidTransition as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        transcript = TranscriptDelta(
            text=request.text.strip(),
            is_final=True,
            confidence=1.0,
            pts_ms=0,
        )
        events: list[StreamEnvelope] = [
            ctx.envelope(
                seq=1,
                kind=EventKind.FINAL_TRANSCRIPT,
                payload={"text": transcript.text, "confidence": transcript.confidence},
            )
        ]

        voice_ready = bool(status_payload["ready"]["voice_avatar"])
        runtime_context = (
            "You are the local Russian assistant running inside ru-local-avatar. "
            "Be precise about current runtime capabilities. "
            + (
                "Current verified state: the full voice pipeline is connected — GigaAM STT, "
                "Qwen text via vLLM, Qwen3-TTS speech, Audio2Face-3D lip-sync, and the VRM "
                "avatar over LiveKit. This text chat endpoint is the fallback path."
                if voice_ready
                else (
                    "Current verified state: only local Qwen text dialogue is connected "
                    "through vLLM in this process. The voice/avatar hot path is not "
                    "available here; do not claim voice, speech synthesis, lip-sync, "
                    "or browser media are connected."
                )
            )
        )
        model = VllmDialogueModel(base_url=config.vllm_base_url, model=config.llm_model)
        try:
            chunks = [
                token.text
                async for token in model.stream(
                    [
                        {"role": "system", "content": runtime_context},
                        {"role": "user", "content": transcript.text},
                    ]
                )
            ]
        except DialogueModelUnavailable as exc:
            session.interrupt("llm_unavailable")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"vLLM dialogue model unavailable: {exc}",
            ) from exc

        assistant_text = "".join(chunks).strip()
        events.append(
            ctx.envelope(
                seq=2,
                kind=EventKind.PARTIAL_TEXT,
                payload={"text": assistant_text, "source": "vllm"},
            )
        )
        session.finish_turn("text_response_complete")
        return TextChatResponse(
            session_id=session.session_id,
            turn_id=ctx.turn_id,
            generation_id=ctx.generation_id,
            branch_state=session.branch_state.value,
            assistant_text=assistant_text,
            events=[event.to_wire() for event in events],
        )

    @app.post("/api/admin/sessions/{session_id}/cancel", response_model=CancelResponse)
    async def cancel_session(session_id: str, request: Request) -> CancelResponse:
        _require_admin_access(config, request)
        session = store.get(session_id)
        session.interrupt("admin_cancel")
        if voice_runtime is not None:
            await voice_runtime.stop_worker(session_id)
        store.release_voice(session_id)
        return CancelResponse(
            session_id=session.session_id,
            generation_id=session.generation_id,
            state=session.branch_state.value,
        )

    @app.get(
        "/api/admin/sessions/{session_id}/conversation",
        response_model=ConversationEventsResponse,
    )
    async def get_conversation_events(
        request: Request,
        session_id: str,
        limit: int = Query(default=200, ge=1, le=500),
    ) -> ConversationEventsResponse:
        _require_admin_access(config, request)
        if voice_runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="voice runtime is unavailable",
            )
        events = voice_runtime.conversation_audit.list_session(session_id, limit=limit)
        return ConversationEventsResponse(session_id=session_id, events=events)

    @app.get(
        "/api/admin/sessions/{session_id}/turns",
        response_model=TurnTraceResponse,
    )
    async def get_turn_traces(
        request: Request,
        session_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> TurnTraceResponse:
        _require_admin_access(config, request)
        if voice_runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="voice runtime is unavailable",
            )
        events = voice_runtime.conversation_audit.list_session(session_id, limit=limit)
        turns = [
            {
                "turn_id": event["turn_id"],
                "generation_id": event["generation_id"],
                "created_at": event["created_at"],
                **event.get("payload", {}),
            }
            for event in events
            if event.get("event_type") == "turn_trace"
        ]
        return TurnTraceResponse(session_id=session_id, turns=turns)

    @app.get("/api/acceptance")
    async def acceptance() -> dict[str, Any]:
        return asdict(thresholds)

    return app


app = create_app()
