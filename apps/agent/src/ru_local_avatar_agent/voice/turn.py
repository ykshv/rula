"""Turn-taking policy.

Pure logic, no torch/audio dependencies, so the exact speculative/EOT/barge-in
behaviour is unit-testable on any host. The policy is fed fixed-size VAD frames
(one speech probability per frame) and emits signals the session worker acts on.

Latency contract (defaults, tuned against measured false-early rates):
- speculative LLM start after ~128 ms of silence,
- EOT commit after ~352 ms of silence when the partial transcript ends with
  terminal punctuation AND has enough words (GigaAM punctuates fragments too
  eagerly for a shorter window), else ~512 ms,
- incomplete Russian tails wait longer before EOT commit,
- barge-in after ~288 ms of stronger sustained voiced audio while the agent speaks,
  but playback code suppresses early speaker-echo right after first audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TurnSignal(StrEnum):
    USER_SPEECH_START = "user_speech_start"
    SPECULATE = "speculate"
    COMMIT = "commit"
    DISCARD = "discard"
    # Two-phase barge-in: DUCK pauses playback almost immediately so the
    # avatar audibly yields; BARGE_IN (longer, verified by transcript in the
    # worker) actually cancels the answer; ABORT un-ducks after a false
    # trigger (click/breath) without dropping the response.
    BARGE_DUCK = "barge_duck"
    BARGE_ABORT = "barge_abort"
    BARGE_IN = "barge_in"


TERMINAL_PUNCTUATION = (".", "!", "?", "…")
INCOMPLETE_TAIL_WORDS = {
    "а",
    "без",
    "в",
    "во",
    "вот",
    "да",
    "для",
    "до",
    "за",
    "и",
    "или",
    "именно",
    "к",
    "как",
    "ко",
    "короче",
    "между",
    "на",
    "над",
    "но",
    "ну",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "про",
    "с",
    "слушай",
    "со",
    "типа",
    "у",
    "через",
    "что",
    "чтобы",
}


@dataclass(frozen=True, slots=True)
class TurnTuning:
    frame_ms: int = 32
    voiced_on_probability: float = 0.60
    # Tight hysteresis: Silero holds elevated probabilities over TTS-style
    # trailing energy, and every extra frame of "voiced" delays EOT 1:1.
    voiced_off_probability: float = 0.50
    min_user_speech_ms: int = 160
    speculative_silence_ms: int = 128
    commit_silence_ms: int = 512
    punctuated_commit_silence_ms: int = 352
    incomplete_commit_silence_ms: int = 1152
    # GigaAM's e2e punctuation happily terminates fragments ("Ты умеешь?"),
    # so the fast path additionally requires a minimum word count. These
    # thresholds trade ~150 ms of EOT for a large false-early reduction;
    # speculation hides most of it from first-audio latency.
    punctuated_commit_min_words: int = 3
    # Barge-in uses a stricter gate than normal user speech start. Browser AEC
    # and room noise can keep Silero around the normal 0.60 threshold during
    # avatar playback; requiring a stronger, longer run prevents clicks,
    # breathing, and TTS echo from cutting off the answer.
    barge_in_probability: float = 0.78
    barge_in_voiced_ms: int = 288
    barge_in_min_speaking_ms: int = 900
    # Duck (pause playback) much earlier than the full barge decision; a
    # false duck costs a ~0.3 s pause, a slow duck costs talking over the
    # user for a second.
    barge_duck_voiced_ms: int = 128
    barge_duck_abort_quiet_ms: int = 192
    max_utterance_ms: int = 25_000


@dataclass(slots=True)
class TurnPolicy:
    tuning: TurnTuning = field(default_factory=TurnTuning)

    _voiced: bool = False
    _utterance_active: bool = False
    _speculated: bool = False
    _barge_signalled: bool = False
    _speech_ms: int = 0
    _silence_ms: int = 0
    _utterance_ms: int = 0
    _voiced_run_ms: int = 0
    _barge_voiced_run_ms: int = 0
    _barge_quiet_ms: int = 0
    _duck_signalled: bool = False
    _partial_terminal: bool = False
    _partial_incomplete: bool = False

    def note_partial(self, text: str) -> None:
        """Record the latest partial transcript for punctuation-aware commits."""
        stripped = text.rstrip()
        self._partial_terminal = stripped.endswith(TERMINAL_PUNCTUATION) and (
            len(stripped.split()) >= self.tuning.punctuated_commit_min_words
        )
        self._partial_incomplete = _ends_in_incomplete_tail(stripped)

    def reset_turn(self) -> None:
        self._utterance_active = False
        self._speculated = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0
        self._voiced_run_ms = 0
        self._barge_voiced_run_ms = 0
        self._barge_signalled = False
        self._partial_terminal = False
        self._partial_incomplete = False

    def reset_barge(self) -> None:
        self._barge_voiced_run_ms = 0
        self._barge_quiet_ms = 0
        self._barge_signalled = False
        self._duck_signalled = False

    @property
    def utterance_active(self) -> bool:
        return self._utterance_active

    def feed(self, speech_probability: float, *, agent_speaking: bool) -> list[TurnSignal]:
        t = self.tuning
        if speech_probability >= t.voiced_on_probability:
            self._voiced = True
        elif speech_probability <= t.voiced_off_probability:
            self._voiced = False
        signals: list[TurnSignal] = []

        if agent_speaking:
            if speech_probability >= t.barge_in_probability:
                self._barge_voiced_run_ms += t.frame_ms
                self._barge_quiet_ms = 0
            else:
                self._barge_voiced_run_ms = 0
                self._barge_signalled = False
                if self._duck_signalled:
                    self._barge_quiet_ms += t.frame_ms
            if (
                self._barge_voiced_run_ms >= t.barge_duck_voiced_ms
                and not self._duck_signalled
            ):
                self._duck_signalled = True
                signals.append(TurnSignal.BARGE_DUCK)
            if (
                self._barge_voiced_run_ms >= t.barge_in_voiced_ms
                and not self._barge_signalled
            ):
                self._barge_signalled = True
                signals.append(TurnSignal.BARGE_IN)
            if (
                self._duck_signalled
                and not self._barge_signalled
                and self._barge_quiet_ms >= t.barge_duck_abort_quiet_ms
            ):
                self._duck_signalled = False
                self._barge_quiet_ms = 0
                signals.append(TurnSignal.BARGE_ABORT)
            # While the agent is speaking the user's turn state does not
            # advance; barge-in resets the pipeline and the frames that follow
            # (with agent_speaking=False) start the new utterance.
            return signals

        self._barge_voiced_run_ms = 0
        self._barge_quiet_ms = 0
        self._barge_signalled = False
        self._duck_signalled = False

        if self._voiced:
            self._voiced_run_ms += t.frame_ms
            if self._utterance_active and self._speculated and self._silence_ms > 0:
                # User resumed after a pause we speculated on.
                self._speculated = False
                signals.append(TurnSignal.DISCARD)
            if not self._utterance_active:
                self._utterance_active = True
                self._speech_ms = 0
                self._utterance_ms = 0
                signals.append(TurnSignal.USER_SPEECH_START)
            self._speech_ms += t.frame_ms
            self._silence_ms = 0
            self._utterance_ms += t.frame_ms
            if self._utterance_ms >= t.max_utterance_ms:
                signals.append(self._commit())
            return signals

        self._voiced_run_ms = 0

        if not self._utterance_active:
            return signals

        self._silence_ms += t.frame_ms
        self._utterance_ms += t.frame_ms
        if self._speech_ms < t.min_user_speech_ms:
            # A blip too short to be speech; drop the utterance once silence
            # outlasts the speculative window.
            if self._silence_ms >= t.speculative_silence_ms:
                self.reset_turn()
            return signals

        if self._partial_incomplete:
            commit_after = max(t.commit_silence_ms, t.incomplete_commit_silence_ms)
        elif self._partial_terminal:
            commit_after = t.punctuated_commit_silence_ms
        else:
            commit_after = t.commit_silence_ms
        if self._silence_ms >= commit_after:
            signals.append(self._commit())
            return signals
        if not self._speculated and self._silence_ms >= t.speculative_silence_ms:
            self._speculated = True
            signals.append(TurnSignal.SPECULATE)
        return signals

    def _commit(self) -> TurnSignal:
        self.reset_turn()
        return TurnSignal.COMMIT


def _ends_in_incomplete_tail(text: str) -> bool:
    normalized = text.strip().lower().rstrip(",;:…")
    if not normalized or normalized.endswith(TERMINAL_PUNCTUATION):
        return False
    words = normalized.split()
    if not words:
        return False
    last = words[-1].strip("«»\"'()[]{}.,!?;:…")
    if last in INCOMPLETE_TAIL_WORDS:
        return True
    if len(words) >= 2:
        tail = " ".join(word.strip("«»\"'()[]{}.,!?;:…") for word in words[-2:])
        return tail == "потому что"
    return False
