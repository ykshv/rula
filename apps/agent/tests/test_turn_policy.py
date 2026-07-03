from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.turn import TurnPolicy, TurnSignal, TurnTuning

TUNING = TurnTuning()
FRAME = TUNING.frame_ms

VOICED = 0.95
SILENT = 0.05


def feed_ms(policy: TurnPolicy, probability: float, ms: int, *, agent_speaking: bool = False):
    signals = []
    for _ in range(ms // FRAME):
        signals.extend(policy.feed(probability, agent_speaking=agent_speaking))
    return signals


class TurnPolicyTest(unittest.TestCase):
    def test_speech_start_then_speculate_then_commit(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(policy, VOICED, 800)
        self.assertEqual(signals, [TurnSignal.USER_SPEECH_START])

        signals = feed_ms(policy, SILENT, TUNING.speculative_silence_ms)
        self.assertEqual(signals, [TurnSignal.SPECULATE])

        signals = feed_ms(policy, SILENT, TUNING.commit_silence_ms)
        self.assertEqual(signals, [TurnSignal.COMMIT])
        self.assertFalse(policy.utterance_active)

    def test_punctuated_partial_commits_faster(self) -> None:
        policy = TurnPolicy()
        feed_ms(policy, VOICED, 800)
        policy.note_partial("Привет, как дела?")
        signals = feed_ms(policy, SILENT, TUNING.punctuated_commit_silence_ms)
        self.assertIn(TurnSignal.COMMIT, signals)

    def test_incomplete_russian_tail_delays_commit(self) -> None:
        policy = TurnPolicy()
        feed_ms(policy, VOICED, 800)
        policy.note_partial("Посмотри логи последней сессии. Именно по")

        signals = feed_ms(policy, SILENT, TUNING.commit_silence_ms)

        self.assertIn(TurnSignal.SPECULATE, signals)
        self.assertNotIn(TurnSignal.COMMIT, signals)

        signals = feed_ms(policy, SILENT, TUNING.incomplete_commit_silence_ms)
        self.assertIn(TurnSignal.COMMIT, signals)

    def test_terminal_russian_sentence_is_not_incomplete_tail(self) -> None:
        policy = TurnPolicy()
        feed_ms(policy, VOICED, 800)
        policy.note_partial("Потому что явно лагает.")

        signals = feed_ms(policy, SILENT, TUNING.punctuated_commit_silence_ms)

        self.assertIn(TurnSignal.COMMIT, signals)

    def test_resumed_speech_discards_speculation(self) -> None:
        policy = TurnPolicy()
        feed_ms(policy, VOICED, 800)
        signals = feed_ms(policy, SILENT, TUNING.speculative_silence_ms)
        self.assertEqual(signals, [TurnSignal.SPECULATE])

        signals = feed_ms(policy, VOICED, 200)
        self.assertEqual(signals[0], TurnSignal.DISCARD)
        # Utterance continues, no new USER_SPEECH_START.
        self.assertNotIn(TurnSignal.USER_SPEECH_START, signals)

        # A later pause speculates again, then commits.
        signals = feed_ms(policy, SILENT, TUNING.commit_silence_ms + FRAME)
        self.assertEqual(
            [s for s in signals if s != TurnSignal.DISCARD],
            [TurnSignal.SPECULATE, TurnSignal.COMMIT],
        )

    def test_short_blip_is_dropped(self) -> None:
        policy = TurnPolicy()
        feed_ms(policy, VOICED, FRAME * 2)  # 64 ms blip < min_user_speech_ms
        signals = feed_ms(policy, SILENT, TUNING.commit_silence_ms * 2)
        self.assertNotIn(TurnSignal.COMMIT, signals)
        self.assertNotIn(TurnSignal.SPECULATE, signals)
        self.assertFalse(policy.utterance_active)

    def test_barge_in_during_agent_speech(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(policy, VOICED, TUNING.barge_in_voiced_ms, agent_speaking=True)
        # Two-phase: duck fires early, the full barge signal follows.
        self.assertEqual(signals, [TurnSignal.BARGE_DUCK, TurnSignal.BARGE_IN])
        # Sustained speech does not re-signal.
        signals = feed_ms(policy, VOICED, 500, agent_speaking=True)
        self.assertEqual(signals, [])

    def test_short_noise_ducks_then_aborts_without_barge(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(
            policy,
            VOICED,
            TUNING.barge_in_voiced_ms - FRAME,
            agent_speaking=True,
        )
        self.assertEqual(signals, [TurnSignal.BARGE_DUCK])
        self.assertNotIn(TurnSignal.BARGE_IN, signals)
        # Noise stops before the barge threshold: playback must resume.
        signals = feed_ms(
            policy,
            SILENT,
            TUNING.barge_duck_abort_quiet_ms + FRAME,
            agent_speaking=True,
        )
        self.assertEqual(signals, [TurnSignal.BARGE_ABORT])

    def test_sub_duck_noise_emits_nothing(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(
            policy,
            VOICED,
            TUNING.barge_duck_voiced_ms - FRAME,
            agent_speaking=True,
        )
        self.assertEqual(signals, [])

    def test_barge_in_uses_stricter_probability_than_speech_start(self) -> None:
        policy = TurnPolicy()
        moderate_voice_probability = TUNING.voiced_on_probability + 0.05

        signals = feed_ms(
            policy,
            moderate_voice_probability,
            TUNING.barge_in_voiced_ms * 2,
            agent_speaking=True,
        )

        self.assertEqual(signals, [])

    def test_no_barge_in_when_user_silent(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(policy, SILENT, 1000, agent_speaking=True)
        self.assertEqual(signals, [])

    def test_max_utterance_forces_commit(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(policy, VOICED, TUNING.max_utterance_ms + FRAME * 4)
        self.assertIn(TurnSignal.COMMIT, signals)

    def test_barge_in_speech_becomes_new_utterance(self) -> None:
        policy = TurnPolicy()
        feed_ms(policy, VOICED, TUNING.barge_in_voiced_ms, agent_speaking=True)
        # Worker flushes playback; agent no longer speaking.
        signals = feed_ms(policy, VOICED, 400, agent_speaking=False)
        self.assertIn(TurnSignal.USER_SPEECH_START, signals)
        signals = feed_ms(policy, SILENT, TUNING.commit_silence_ms)
        self.assertIn(TurnSignal.COMMIT, signals)

    def test_reset_barge_allows_later_barge_signal(self) -> None:
        policy = TurnPolicy()
        signals = feed_ms(policy, VOICED, TUNING.barge_in_voiced_ms, agent_speaking=True)
        self.assertEqual(signals, [TurnSignal.BARGE_DUCK, TurnSignal.BARGE_IN])

        policy.reset_barge()
        signals = feed_ms(policy, VOICED, TUNING.barge_in_voiced_ms, agent_speaking=True)

        self.assertEqual(signals, [TurnSignal.BARGE_DUCK, TurnSignal.BARGE_IN])


if __name__ == "__main__":
    unittest.main()
