from __future__ import annotations

from dataclasses import dataclass

from ru_local_avatar_agent.domain.contracts import (
    AvatarTimeline,
    DialogueModel,
    FaceAnimator,
    SpeechSynthesizer,
    TranscriptDelta,
    TurnDetector,
)
from ru_local_avatar_agent.domain.events import BranchState, EventKind, StreamEnvelope
from ru_local_avatar_agent.domain.session import SessionStateMachine
from ru_local_avatar_agent.runtime.clause_chunker import ClauseChunker


@dataclass(slots=True)
class PipelineResult:
    envelopes: list[StreamEnvelope]
    speculative_started: bool
    committed: bool


class InMemoryTimeline(AvatarTimeline):
    def __init__(self) -> None:
        self.events: list[StreamEnvelope] = []

    async def emit(self, envelope: StreamEnvelope) -> None:
        self.events.append(envelope)


class ConversationPipeline:
    def __init__(
        self,
        *,
        turn_detector: TurnDetector,
        dialogue_model: DialogueModel,
        synthesizer: SpeechSynthesizer,
        face_animator: FaceAnimator,
        timeline: AvatarTimeline,
        token_target: int = 12,
    ) -> None:
        self.turn_detector = turn_detector
        self.dialogue_model = dialogue_model
        self.synthesizer = synthesizer
        self.face_animator = face_animator
        self.timeline = timeline
        self.token_target = token_target

    async def run_text_turn(
        self,
        *,
        state: SessionStateMachine,
        transcript: TranscriptDelta,
    ) -> PipelineResult:
        decision = await self.turn_detector.decide(transcript=transcript, recent_audio=None)
        speculative_started = False
        envelopes: list[StreamEnvelope] = []

        if decision.should_speculate and state.branch_state == BranchState.LISTENING:
            ctx = state.start_speculative(decision.reason)
            speculative_started = True
            event = ctx.envelope(
                seq=0,
                kind=EventKind.AVATAR_STATE,
                payload={"state": "thinking", "shift_probability": decision.shift_probability},
            )
            await self.timeline.emit(event)
            envelopes.append(event)

        if decision.should_commit:
            ctx = state.commit_eot(decision.reason)
        elif state.branch_state == BranchState.SPECULATIVE:
            ctx = state.commit_eot("speculative_commit_for_mock_pipeline")
        else:
            return PipelineResult(envelopes=envelopes, speculative_started=False, committed=False)

        transcript_event = ctx.envelope(
            seq=1,
            kind=EventKind.FINAL_TRANSCRIPT,
            payload={"text": transcript.text, "confidence": transcript.confidence},
        )
        await self.timeline.emit(transcript_event)
        envelopes.append(transcript_event)

        chunker = ClauseChunker(token_target=self.token_target)
        seq = 2
        async for token in self.dialogue_model.stream([{"role": "user", "content": transcript.text}]):
            if token.text:
                partial = ctx.envelope(
                    seq=seq,
                    kind=EventKind.PARTIAL_TEXT,
                    payload={"text": token.text},
                )
                await self.timeline.emit(partial)
                envelopes.append(partial)
                seq += 1
            chunk = chunker.push(token.text)
            if chunk:
                speech_event = ctx.envelope(
                    seq=seq,
                    kind=EventKind.SPEECH_SEGMENT,
                    payload={"text": chunk.text, "chunk_reason": chunk.reason},
                )
                await self.timeline.emit(speech_event)
                envelopes.append(speech_event)
                seq += 1

        final_chunk = chunker.finish()
        if final_chunk:
            final_event = ctx.envelope(
                seq=seq,
                kind=EventKind.SPEECH_SEGMENT,
                payload={"text": final_chunk.text, "chunk_reason": final_chunk.reason},
            )
            await self.timeline.emit(final_event)
            envelopes.append(final_event)

        state.start_speaking()
        return PipelineResult(
            envelopes=envelopes,
            speculative_started=speculative_started,
            committed=True,
        )
