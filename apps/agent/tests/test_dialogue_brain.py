from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.brain import (
    MACHINE_CHALLENGE_REPLY,
    ConversationBrain,
    ConversationState,
    DialogueBrain,
    ResponsePlanner,
    build_messages,
)


class DialogueBrainTest(unittest.TestCase):
    def test_machine_challenge_uses_scripted_meme_reply(self) -> None:
        # Product requirement: being called a machine/robot gets the fixed
        # meme line, verbatim.
        brain = DialogueBrain()

        for phrase in (
            "Ты же всего лишь машина, что ты можешь?",
            "Да ты просто робот.",
            "Ты машина?",
        ):
            reply = brain.direct_response(phrase)
            self.assertEqual(reply, MACHINE_CHALLENGE_REPLY, phrase)

    def test_machine_words_in_other_contexts_do_not_trigger_meme(self) -> None:
        brain = DialogueBrain()
        for phrase in (
            "У меня сломалась машина, что делать?",
            "Расскажи про роботов Boston Dynamics.",
            "Как работает нейросеть?",
        ):
            self.assertNotEqual(
                brain.direct_response(phrase), MACHINE_CHALLENGE_REPLY, phrase
            )

    def test_prompt_keeps_voice_context_and_recent_history(self) -> None:
        brain = DialogueBrain()
        brain.record_turn("Мы обсуждали голос аватара.", "Да, голос дребезжал на склейках.")

        messages = brain.build_messages(
            [{"role": "user", "content": "Продолжим про качество речи."}],
            "Что дальше делать?",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("живой голосовой разговор", messages[0]["content"])
        self.assertIn("Мы обсуждали голос аватара", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Что дальше делать?"})

    def test_committed_user_turn_survives_interrupted_assistant_answer(self) -> None:
        brain = DialogueBrain()
        brain.record_user_turn("Как тебя зовут?")

        messages = brain.build_messages([], "А что я только что спросил?")
        system_prompt = messages[0]["content"]

        self.assertIn("Как тебя зовут?", system_prompt)
        self.assertNotIn(
            "1. А что я только что спросил?",
            system_prompt,
        )

    def test_current_user_turn_is_not_used_as_its_own_context(self) -> None:
        brain = DialogueBrain()
        brain.record_user_turn("Как тебя зовут?")
        brain.record_user_turn("А что я только что спросил?")

        messages = brain.build_messages([], "А что я только что спросил?")
        system_prompt = messages[0]["content"]

        self.assertIn("Как тебя зовут?", system_prompt)
        self.assertNotIn(
            "А что я только что спросил?.",
            system_prompt,
        )

    def test_memory_recall_answers_previous_user_question_directly(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Как тебя зовут?")
        brain.record_user_turn("А что я только что спросил?")

        plan = brain.plan_response([], "А что я только что спросил?")

        self.assertEqual(plan.mode, "direct")
        self.assertIn("Как тебя зовут?", plan.direct_text)
        self.assertEqual(plan.messages, [])

    def test_previous_recall_answers_turn_before_last(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Ты же всего лишь машина, что ты можешь?")
        brain.record_user_turn("А что я только что спросил?")
        brain.record_user_turn("А до этого?")

        plan = brain.plan_response([], "А до этого?")

        self.assertEqual(plan.mode, "direct")
        self.assertIn("Ты же всего лишь машина", plan.direct_text)

    def test_called_as_recall_uses_structured_state(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Ты же всего лишь машина, что ты можешь?")

        plan = brain.plan_response([], "А как я тебя называл?")

        self.assertEqual(plan.mode, "direct")
        self.assertIn("машиной", plan.direct_text)

    def test_default_identity_uses_wired_avatar_name(self) -> None:
        brain = ConversationBrain()

        plan = brain.plan_response([], "Как тебя зовут?")

        self.assertEqual(plan.mode, "direct")
        self.assertEqual(plan.direct_text, "Меня зовут Даздраперма.")

    def test_name_negotiation_keeps_candidate_name_for_identity(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Почему Лена?")
        reject = brain.plan_response([], "Нет, не нравится.")
        brain.record_user_turn("Нет, не нравится.")
        propose = brain.plan_response([], "Может, ты Даздраперма?")
        brain.record_user_turn("Может, ты Даздраперма?")
        identity = brain.plan_response([], "Да как тебя зовут?")

        self.assertEqual(reject.mode, "direct")
        self.assertEqual(propose.direct_text, "Можно, Даздраперма звучит лучше.")
        self.assertEqual(identity.direct_text, "Меня зовут Даздраперма.")

    def test_normal_dialogue_path_builds_llm_messages_with_state(self) -> None:
        state = ConversationState()
        state.user_called_avatar_as.append("машиной")
        planner = ResponsePlanner()

        plan = planner.plan("Расскажи что-нибудь умное.", state, [])

        self.assertEqual(plan.mode, "llm")
        self.assertIn("Пользователь называл тебя: машиной.", plan.messages[0]["content"])

    def test_interrupted_assistant_answer_does_not_become_final(self) -> None:
        brain = ConversationBrain()

        brain.record_assistant_interrupted("Я хотела сказать, что...")

        self.assertEqual(brain.state.last_assistant_interrupted, "Я хотела сказать, что...")
        self.assertEqual(brain.state.last_assistant_final, "")

    def test_compat_build_messages_uses_brain_prompt(self) -> None:
        messages = build_messages([], "Привет")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("digital human", messages[0]["content"])
        self.assertIn("чем могу помочь", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Привет"})


if __name__ == "__main__":
    unittest.main()
