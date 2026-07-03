from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.tts import estimate_max_new_tokens


class TtsBudgetTest(unittest.TestCase):
    def test_short_clause_does_not_force_multi_second_budget(self) -> None:
        self.assertEqual(estimate_max_new_tokens("Готово."), 18)

    def test_budget_scales_with_text_length(self) -> None:
        budget = estimate_max_new_tokens("Мне нравится разговаривать с тобой по-русски.")

        self.assertGreater(budget, 18)
        self.assertLess(budget, 96)

    def test_budget_is_capped(self) -> None:
        budget = estimate_max_new_tokens("а" * 1000, max_new_tokens=128)

        self.assertEqual(budget, 128)


if __name__ == "__main__":
    unittest.main()
