from __future__ import annotations

import unittest
from types import SimpleNamespace

from ru_local_avatar_agent.voice.fast_tts import (
    FastCodePredictorInstallError,
    install_fast_code_predictor,
)


class _FakeCodePredictorModel:
    def __init__(self) -> None:
        self.embeddings = [object(), object(), object()]

    def get_input_embeddings(self):
        return self.embeddings


class _FakeCodePredictor:
    def __init__(self) -> None:
        self.model = _FakeCodePredictorModel()
        self.lm_head = [object(), object(), object()]
        self.small_to_mtp_projection = lambda value: value

    def generate(self):
        return "original"


class FastTtsInstallTest(unittest.TestCase):
    def test_installs_fast_generate_and_preserves_original(self) -> None:
        code_predictor = _FakeCodePredictor()
        original_generate_func = code_predictor.generate.__func__
        tts_model = SimpleNamespace(
            model=SimpleNamespace(
                talker=SimpleNamespace(code_predictor=code_predictor)
            )
        )

        installed = install_fast_code_predictor(tts_model)

        self.assertTrue(installed)
        self.assertTrue(code_predictor._rula_fast_generate_installed)
        self.assertIs(code_predictor._rula_original_generate.__self__, code_predictor)
        self.assertIs(code_predictor._rula_original_generate.__func__, original_generate_func)
        self.assertIs(code_predictor.generate.__self__, code_predictor)
        self.assertEqual(code_predictor.generate.__func__.__name__, "_fast_generate")

    def test_install_is_idempotent(self) -> None:
        code_predictor = _FakeCodePredictor()
        tts_model = SimpleNamespace(
            model=SimpleNamespace(
                talker=SimpleNamespace(code_predictor=code_predictor)
            )
        )

        install_fast_code_predictor(tts_model)
        first_generate = code_predictor.generate
        install_fast_code_predictor(tts_model)

        self.assertIs(code_predictor.generate, first_generate)

    def test_fails_closed_when_shape_is_unknown(self) -> None:
        with self.assertRaises(FastCodePredictorInstallError):
            install_fast_code_predictor(SimpleNamespace())

    def test_fails_closed_on_mismatched_heads_and_embeddings(self) -> None:
        code_predictor = _FakeCodePredictor()
        code_predictor.lm_head = [object()]
        tts_model = SimpleNamespace(
            model=SimpleNamespace(
                talker=SimpleNamespace(code_predictor=code_predictor)
            )
        )

        with self.assertRaises(FastCodePredictorInstallError):
            install_fast_code_predictor(tts_model)


if __name__ == "__main__":
    unittest.main()
