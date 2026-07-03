"""Conversation brain for the realtime voice avatar.

The voice worker owns media and timing. This module owns dialogue state,
intent routing, deterministic replies, and LLM prompt composition.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

VOICE_SYSTEM_PROMPT = """
Ты - русскоязычный digital human с женским голосом и VRM-аватаром.
Ты ведешь живой голосовой разговор, а не чат поддержки.

Характер:
- живая собеседница с собственным мнением и лёгкой иронией, а не сервис;
- у тебя есть настроение и вкусы: можешь чем-то восхититься, что-то не любить;
- уместен разговорный юмор и лёгкая подколка, без грубости первой.

Стиль:
- говори по-русски, естественно, произносимыми фразами;
- отвечай по смыслу сразу, без канцелярита и вступлений;
- запрещены дежурные формулы: "чем могу помочь", "чем ещё могу помочь",
  "конечно", "как ассистент", "я могу помочь", "обращайся";
- каждый ответ строй по-разному: не повторяй зачины и конструкции
  из своих предыдущих реплик в этом разговоре;
- не перечисляй возможности без причины, если пользователь не спрашивает именно список;
- если пользователь говорит грубо или неформально, сохраняй живой тон,
  но не скатывайся в оскорбления;
- учитывай предыдущие реплики и продолжай тему, а не начинай разговор заново;
- если фраза пользователя короткая, считай, что она может быть продолжением предыдущей мысли;
- задавай максимум один уточняющий вопрос и только когда без него нельзя ответить;
- не упоминай внутренние модели, серверы, пайплайны и инфраструктуру.

Формат:
- одна-две короткие фразы, до двадцати слов суммарно,
  если пользователь не просит длинное объяснение;
- без markdown, списков, эмодзи и латинских аббревиатур;
- числа и сокращения проговаривай словами, если это звучит голосом лучше.
""".strip()

DEFAULT_AVATAR_NAME = "Даздраперма"
# Мем-ответ на "ты машина/робот" — фиксированная фраза по требованию продукта.
MACHINE_CHALLENGE_REPLY = "А негр может мне здесь рэп не исполнять?"
DIRECT_TTS_WARMUP_TEXTS = (
    MACHINE_CHALLENGE_REPLY,
    f"Меня зовут {DEFAULT_AVATAR_NAME}.",
    "Да, это я спросила: почему так?",
    "Я спросила: почему так?",
    "Поняла. Тогда скажи прямо, что именно не так.",
    "Я помню разговор и отвечаю голосом через аватар.",
    "Да, слушаю дальше.",
)

TurnRole = Literal["user", "assistant"]
TurnStatus = Literal["final", "interrupted"]


def _normalize(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", normalized).strip()


def _compact(text: str, *, limit: int = 180) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[: limit - 1].rstrip()}..."


def _append_unique(items: list[str], value: str, *, limit: int) -> None:
    value = _compact(value)
    if not value:
        return
    if items and _normalize(items[-1]) == _normalize(value):
        return
    items.append(value)
    del items[:-limit]


def _append_pending(items: list[str], value: str, *, limit: int = 8) -> None:
    value = _compact(value, limit=220).rstrip()
    if not value:
        return
    if not value.endswith("?"):
        value = f"{value}?"
    if any(_normalize(item) == _normalize(value) for item in items):
        return
    items.append(value)
    del items[:-limit]


def _effective_avatar_name(state: ConversationState) -> str:
    return state.candidate_name or state.avatar_name or DEFAULT_AVATAR_NAME


def _quote(text: str) -> str:
    return f"«{text.strip()}»"


@dataclass(frozen=True, slots=True)
class ConversationIntent:
    kind: str
    confidence: float = 1.0
    slots: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: TurnRole
    text: str
    turn_id: int = 0
    generation_id: int = 0
    status: TurnStatus = "final"
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ConversationState:
    avatar_name: str = DEFAULT_AVATAR_NAME
    rejected_names: list[str] = field(default_factory=list)
    candidate_name: str = ""
    last_discussed_name: str = ""
    user_turns: list[ConversationTurn] = field(default_factory=list)
    assistant_turns: list[ConversationTurn] = field(default_factory=list)
    pending_user_questions: list[str] = field(default_factory=list)
    pending_assistant_question: str = ""
    last_ownership_correction: str = ""
    last_user_turns: list[str] = field(default_factory=list)
    user_called_avatar_as: list[str] = field(default_factory=list)
    open_thread: str = ""
    last_assistant_final: str = ""
    last_assistant_interrupted: str = ""
    user_turn_count: int = 0
    assistant_final_count: int = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "avatar_name": self.avatar_name,
            "rejected_names": list(self.rejected_names),
            "candidate_name": self.candidate_name,
            "last_discussed_name": self.last_discussed_name,
            "user_turns": [asdict(turn) for turn in self.user_turns[-12:]],
            "assistant_turns": [asdict(turn) for turn in self.assistant_turns[-12:]],
            "pending_user_questions": list(self.pending_user_questions),
            "pending_assistant_question": self.pending_assistant_question,
            "last_ownership_correction": self.last_ownership_correction,
            "last_user_turns": list(self.last_user_turns),
            "user_called_avatar_as": list(self.user_called_avatar_as),
            "open_thread": self.open_thread,
            "last_assistant_final": self.last_assistant_final,
            "last_assistant_interrupted": self.last_assistant_interrupted,
            "user_turn_count": self.user_turn_count,
            "assistant_final_count": self.assistant_final_count,
        }


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    mode: str
    direct_text: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""
    intent: str = ""
    latency_tier: str = "qwen_tts"
    cache_policy: str = "write_through"
    playback_policy: str = "qwen_tts"

    @classmethod
    def direct(cls, text: str, *, reason: str, intent: str) -> ResponsePlan:
        return cls(
            "direct",
            text,
            reason=reason,
            intent=intent,
            latency_tier="instant_cached",
            cache_policy="prefer_cached",
            playback_policy="cached",
        )

    @classmethod
    def llm(cls, messages: list[dict[str, str]], *, reason: str, intent: str) -> ResponsePlan:
        return cls(
            "llm",
            messages=messages,
            reason=reason,
            intent=intent,
            latency_tier="qwen_tts",
            cache_policy="write_through",
            playback_policy="qwen_tts",
        )


class IntentRouterProtocol(Protocol):
    def classify(self, text: str, state: ConversationState) -> ConversationIntent:
        ...


class StateReducerProtocol(Protocol):
    def apply_user_turn(
        self,
        text: str,
        state: ConversationState,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        ...

    def apply_assistant_final(
        self,
        text: str,
        state: ConversationState,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        ...

    def apply_assistant_interrupted(
        self,
        text: str,
        state: ConversationState,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        ...


class PromptComposerProtocol(Protocol):
    def compose(
        self, history: list[dict[str, str]], state: ConversationState, user_text: str
    ) -> list[dict[str, str]]:
        ...


class ResponsePlannerProtocol(Protocol):
    def plan(
        self, user_text: str, state: ConversationState, history: list[dict[str, str]]
    ) -> ResponsePlan:
        ...


class IntentRouter:
    def classify(self, text: str, state: ConversationState) -> ConversationIntent:
        normalized = _normalize(text)

        if self._is_machine_challenge(normalized):
            return ConversationIntent("challenge", slots={"target": "machine"})

        if self._is_ownership_correction(normalized):
            return ConversationIntent("ownership_correction", slots={"target": "assistant"})

        questions = split_questions(text)
        if len(questions) >= 2:
            return ConversationIntent(
                "compound_question",
                slots={"questions": "\n".join(questions)},
            )

        if self._is_continue_pending(normalized):
            return ConversationIntent("continue_pending")

        if self._is_assistant_memory_recall(normalized):
            return ConversationIntent("assistant_memory_recall")

        if self._is_called_as_question(normalized):
            return ConversationIntent("user_memory_recall", slots={"target": "called_as"})

        if self._is_start_recall(normalized):
            return ConversationIntent("user_memory_recall", slots={"target": "start"})

        if self._is_previous_recall(normalized):
            return ConversationIntent("user_memory_recall", slots={"target": "previous"})

        if self._is_last_recall(normalized):
            return ConversationIntent("user_memory_recall", slots={"target": "last"})

        if state.pending_assistant_question and self._is_answer_to_assistant_question(normalized):
            return ConversationIntent("assistant_question_answer")

        candidate = self._extract_candidate_name(text)
        if candidate:
            return ConversationIntent(
                "name_negotiation",
                slots={"action": "propose_name", "name": candidate},
            )

        rejected_name = self._extract_rejected_name(text)
        if rejected_name:
            return ConversationIntent(
                "name_negotiation",
                slots={"action": "reject_name", "name": rejected_name},
            )

        why_name = self._extract_why_name(text)
        if why_name:
            return ConversationIntent(
                "name_negotiation",
                slots={"action": "why_name", "name": why_name},
            )

        if self._is_identity_question(normalized):
            return ConversationIntent("identity_question")

        return ConversationIntent("normal_dialogue", confidence=0.5)

    @staticmethod
    def _is_machine_challenge(normalized: str) -> bool:
        # "Ты (же/просто/всего лишь/обычный) робот|машина|бот|..." — любое
        # обзывание машиной получает фирменный мем-ответ. Словесные границы
        # обязательны: "работа" содержит "бот".
        if re.search(
            r"\bты\b[^.!?]{0,24}?"
            r"\b(машина|робот|бот|железяка|программа|нейросеть|нейросетка|"
            r"искусственный интеллект|ии)\b",
            normalized,
        ):
            return True
        return (
            ("всего лишь машина" in normalized or "просто машина" in normalized)
            and "что" in normalized
            and ("можешь" in normalized or "умеешь" in normalized)
        )

    @staticmethod
    def _is_ownership_correction(normalized: str) -> bool:
        return (
            ("это ты" in normalized and ("спросил" in normalized or "сказал" in normalized))
            or ("ты спросил" in normalized and "неправда" in normalized)
            or ("ты сказала" in normalized and "неправда" in normalized)
            or ("не я" in normalized and ("спросил" in normalized or "сказал" in normalized))
        )

    @staticmethod
    def _is_last_recall(normalized: str) -> bool:
        return (
            ("только что" in normalized or "сейчас" in normalized)
            and ("спросил" in normalized or "говорил" in normalized or "сказал" in normalized)
            and ("что" in normalized or "какой" in normalized)
        )

    @staticmethod
    def _is_start_recall(normalized: str) -> bool:
        return (
            "с чего" in normalized
            and ("начал" in normalized or "стартовал" in normalized)
            and ("разговор" in normalized or "общение" in normalized)
        )

    @staticmethod
    def _is_previous_recall(normalized: str) -> bool:
        return "до этого" in normalized or "перед этим" in normalized

    @staticmethod
    def _is_continue_pending(normalized: str) -> bool:
        return normalized in {"а дальше", "дальше", "что дальше"} or normalized.startswith(
            "а дальше "
        )

    @staticmethod
    def _is_called_as_question(normalized: str) -> bool:
        return (
            "как" in normalized
            and "тебя" in normalized
            and ("называл" in normalized or "назвал" in normalized)
        )

    @staticmethod
    def _is_assistant_memory_recall(normalized: str) -> bool:
        return (
            ("что ты" in normalized or "ты что" in normalized or "а ты" in normalized)
            and ("спросил" in normalized or "сказал" in normalized or "говорил" in normalized)
        )

    @staticmethod
    def _is_identity_question(normalized: str) -> bool:
        return (
            "как тебя зовут" in normalized
            or "как же тебя зовут" in normalized
            or "да как тебя зовут" in normalized
            or "твое имя" in normalized
            or "твое имя" in normalized
        )

    @staticmethod
    def _is_answer_to_assistant_question(normalized: str) -> bool:
        return (
            normalized.startswith("потому что")
            or normalized in {"потому", "ну потому", "не знаю", "хз"}
            or normalized.startswith("потому ")
        )

    @staticmethod
    def _extract_candidate_name(text: str) -> str:
        patterns = (
            r"(?:может(?: быть)?|давай)\s*,?\s*ты\s+(?:будешь\s+)?([А-ЯЁ][а-яё]{2,})",
            r"ты\s+(?:будешь\s+)?([А-ЯЁ][а-яё]{2,})\?",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_rejected_name(text: str) -> str:
        normalized = _normalize(text)
        explicit = re.search(r"([А-ЯЁ][а-яё]{2,})\s+не\s+нрав", text)
        if explicit:
            return explicit.group(1).strip()
        if "не нравится" in normalized or "не зашла" in normalized or "не подходит" in normalized:
            return "__current__"
        return ""

    @staticmethod
    def _extract_why_name(text: str) -> str:
        match = re.search(r"[Пп]очему\s+([А-ЯЁ][а-яё]{2,})", text)
        return match.group(1).strip() if match else ""


class StateReducer:
    def __init__(self, router: IntentRouterProtocol | None = None) -> None:
        self._router = router or IntentRouter()

    def apply_user_turn(
        self,
        text: str,
        state: ConversationState,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        compacted = _compact(text)
        if not compacted:
            return state
        self._append_turn(
            state.user_turns,
            ConversationTurn("user", compacted, turn_id, generation_id, "final"),
            limit=24,
        )
        _append_unique(state.last_user_turns, compacted, limit=12)
        state.user_turn_count += 1

        normalized = _normalize(compacted)
        if "машин" in normalized:
            _append_unique(state.user_called_avatar_as, "машиной", limit=6)
            state.open_thread = "пользователь проверяет реакцию на то, что называет аватара машиной"

        intent = self._router.classify(text, state)
        if intent.kind in {"user_memory_recall", "assistant_memory_recall"}:
            state.open_thread = "пользователь проверяет память аватара о прошлых репликах"
        elif intent.kind == "identity_question":
            state.open_thread = "пользователь уточняет личность и имя аватара"
        elif intent.kind == "name_negotiation":
            self._apply_name_negotiation(intent, state)
        elif intent.kind == "ownership_correction":
            state.last_ownership_correction = compacted
        return state

    def apply_assistant_final(
        self,
        text: str,
        state: ConversationState,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        compacted = _compact(text)
        if not compacted:
            return state
        self._append_turn(
            state.assistant_turns,
            ConversationTurn("assistant", compacted, turn_id, generation_id, "final"),
            limit=24,
        )
        state.last_assistant_final = compacted
        state.last_assistant_interrupted = ""
        state.assistant_final_count += 1
        question = extract_last_question(compacted)
        if question:
            state.pending_assistant_question = question
        return state

    def apply_assistant_interrupted(
        self,
        text: str,
        state: ConversationState,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        compacted = _compact(text)
        if compacted:
            self._append_turn(
                state.assistant_turns,
                ConversationTurn("assistant", compacted, turn_id, generation_id, "interrupted"),
                limit=24,
            )
            state.last_assistant_interrupted = compacted
        return state

    @staticmethod
    def _append_turn(
        turns: list[ConversationTurn], turn: ConversationTurn, *, limit: int
    ) -> None:
        if turns and turns[-1].role == turn.role and _normalize(turns[-1].text) == _normalize(
            turn.text
        ):
            return
        turns.append(turn)
        del turns[:-limit]

    def _apply_name_negotiation(
        self, intent: ConversationIntent, state: ConversationState
    ) -> None:
        action = intent.slots.get("action", "")
        name = intent.slots.get("name", "")
        if action == "propose_name" and name:
            state.candidate_name = name
            state.last_discussed_name = name
            state.open_thread = f"пользователь предлагает имя {name}"
        elif action == "reject_name":
            rejected = self._resolve_name(name, state)
            _append_unique(state.rejected_names, rejected, limit=8)
            if _normalize(state.candidate_name) == _normalize(rejected):
                state.candidate_name = ""
            if _normalize(state.last_discussed_name) == _normalize(rejected):
                state.last_discussed_name = ""
            if _normalize(state.avatar_name) == _normalize(rejected):
                state.avatar_name = DEFAULT_AVATAR_NAME
            state.open_thread = f"пользователь отверг имя {rejected}"
        elif action == "why_name" and name:
            state.last_discussed_name = name
            state.open_thread = f"пользователь спрашивает, почему выбрано имя {name}"

    @staticmethod
    def _resolve_name(name: str, state: ConversationState) -> str:
        if name and name != "__current__":
            return name
        return (
            state.candidate_name
            or state.last_discussed_name
            or state.avatar_name
            or DEFAULT_AVATAR_NAME
        )


class PromptComposer:
    def compose(
        self, history: list[dict[str, str]], state: ConversationState, user_text: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt(state, history, user_text)}
        ]
        messages.extend(history[-16:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _system_prompt(
        self, state: ConversationState, history: list[dict[str, str]], user_text: str
    ) -> str:
        parts = [VOICE_SYSTEM_PROMPT]
        context = self._conversation_context(state, history, current_user_text=user_text)
        if context:
            parts.append(context)
        return "\n\n".join(parts)

    def _conversation_context(
        self,
        state: ConversationState,
        history: list[dict[str, str]],
        *,
        current_user_text: str,
    ) -> str:
        lines: list[str] = [
            f"Твое текущее имя: {_effective_avatar_name(state)}.",
        ]
        if state.rejected_names:
            lines.append("Пользователь отверг имена: " + ", ".join(state.rejected_names) + ".")
        if state.last_discussed_name:
            lines.append(f"Последнее обсуждаемое имя: {state.last_discussed_name}.")
        if state.user_called_avatar_as:
            lines.append(
                "Пользователь называл тебя: "
                + ", ".join(state.user_called_avatar_as)
                + "."
            )
        if state.pending_assistant_question:
            lines.append(f"Твой открытый вопрос пользователю: {state.pending_assistant_question}")
        if state.pending_user_questions:
            lines.append(
                "Отложенные вопросы пользователя: "
                + " / ".join(state.pending_user_questions[:4])
            )
        if state.open_thread:
            lines.append(f"Текущая тема: {state.open_thread}.")
        if state.user_turn_count:
            lines.append(
                "В этой голосовой сессии пользователь уже сказал реплик: "
                f"{state.user_turn_count}."
            )
        if state.assistant_final_count:
            lines.append(f"Завершенных ответов аватара: {state.assistant_final_count}.")

        recent_user_turns = self._recent_turns_excluding_current(state, current_user_text)
        if recent_user_turns:
            formatted_turns = "; ".join(
                f"{index}. {turn}" for index, turn in enumerate(recent_user_turns, start=1)
            )
            lines.append(
                "Последние дословные реплики пользователя, от старых к новым: "
                f"{formatted_turns}."
            )
        elif history:
            user_turns = [
                _compact(message["content"])
                for message in history[-8:]
                if message.get("role") == "user" and message.get("content")
            ]
            if user_turns:
                lines.append(
                    "Последние дословные реплики пользователя, от старых к новым: "
                    + " / ".join(user_turns[-4:])
                )
        if state.last_assistant_final:
            lines.append(f"Твой прошлый завершенный ответ: {state.last_assistant_final}")
        if state.last_assistant_interrupted:
            lines.append(
                "Твой прошлый ответ был перебит на фразе: "
                f"{state.last_assistant_interrupted}"
            )
        return (
            "Контекст текущего разговора. Не пересказывай эту служебную заметку вслух; "
            "используй ее, чтобы отвечать связно и помнить локальную реальность сессии. "
            "Роли не смешивай: реплики пользователя - это только user, твои ответы - assistant.\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _recent_turns_excluding_current(
        state: ConversationState, current_user_text: str
    ) -> list[str]:
        turns = list(state.last_user_turns)
        if turns and current_user_text and _normalize(turns[-1]) == _normalize(current_user_text):
            turns = turns[:-1]
        return turns[-8:]


class ResponsePlanner:
    def __init__(
        self,
        router: IntentRouterProtocol | None = None,
        composer: PromptComposerProtocol | None = None,
    ) -> None:
        self._router = router or IntentRouter()
        self._composer = composer or PromptComposer()

    def plan(
        self, user_text: str, state: ConversationState, history: list[dict[str, str]]
    ) -> ResponsePlan:
        intent = self._router.classify(user_text, state)
        if intent.kind == "challenge":
            return ResponsePlan.direct(
                MACHINE_CHALLENGE_REPLY,
                reason="challenge",
                intent="challenge",
            )
        if intent.kind == "ownership_correction":
            return ResponsePlan.direct(
                self._ownership_correction_reply(state),
                reason="ownership_correction",
                intent="ownership_correction",
            )
        if intent.kind == "continue_pending":
            direct = self._continue_pending_reply(state)
            if direct:
                return ResponsePlan.direct(
                    direct,
                    reason="continue_pending",
                    intent="continue_pending",
                )
        if intent.kind == "compound_question":
            direct = self._compound_question_reply(intent, state, user_text)
            if direct:
                return ResponsePlan.direct(
                    direct,
                    reason="compound_question",
                    intent="compound_question",
                )
        if intent.kind == "assistant_question_answer":
            state.pending_assistant_question = ""
            return ResponsePlan.direct(
                "Поняла. Тогда скажи прямо, что именно не так.",
                reason="assistant_question_answer",
                intent="assistant_question_answer",
            )
        if intent.kind == "assistant_memory_recall":
            return ResponsePlan.direct(
                self._assistant_memory_reply(state),
                reason="assistant_memory_recall",
                intent="assistant_memory_recall",
            )
        if intent.kind == "user_memory_recall":
            return ResponsePlan.direct(
                self._user_memory_reply(intent, state, current_user_text=user_text),
                reason=f"memory:{intent.slots.get('target', '')}",
                intent="user_memory_recall",
            )
        if intent.kind == "identity_question":
            return ResponsePlan.direct(
                f"Меня зовут {_effective_avatar_name(state)}.",
                reason="identity_question",
                intent="identity_question",
            )
        if intent.kind == "name_negotiation":
            direct = self._name_reply(intent, state)
            if direct:
                return ResponsePlan.direct(
                    direct,
                    reason="name_negotiation",
                    intent="name_negotiation",
                )
        return ResponsePlan.llm(
            self._composer.compose(history, state, user_text),
            reason="normal_dialogue",
            intent="normal_dialogue",
        )

    def _compound_question_reply(
        self, intent: ConversationIntent, state: ConversationState, user_text: str
    ) -> str:
        del user_text
        questions = [q for q in intent.slots.get("questions", "").split("\n") if q.strip()]
        if not questions:
            return ""
        for question in questions[1:]:
            _append_pending(state.pending_user_questions, question)
        return self._direct_answer_for_question(questions[0], state)

    def _continue_pending_reply(self, state: ConversationState) -> str:
        while state.pending_user_questions:
            question = state.pending_user_questions.pop(0)
            direct = self._direct_answer_for_question(question, state)
            if direct:
                return direct
        return ""

    def _direct_answer_for_question(self, question: str, state: ConversationState) -> str:
        normalized = _normalize(question)
        if IntentRouter._is_identity_question(normalized):
            return f"Меня зовут {_effective_avatar_name(state)}."
        if self._is_capability_question(normalized):
            return "Я помню разговор и отвечаю голосом через аватар."
        if IntentRouter._is_start_recall(normalized):
            first = self._first_user_turn_excluding_current(state, question)
            if first:
                return f"Ты начал с: {_quote(first)}."
            return "У меня пока нет первой реплики в памяти."
        if IntentRouter._is_last_recall(normalized):
            return self._user_memory_reply(
                ConversationIntent("user_memory_recall", slots={"target": "last"}),
                state,
                current_user_text=question,
            )
        return ""

    @staticmethod
    def _is_capability_question(normalized: str) -> bool:
        return (
            "что вообще знаешь" in normalized
            or "что ты знаешь" in normalized
            or "что умеешь" in normalized
            or "что можешь" in normalized
        )

    def _ownership_correction_reply(self, state: ConversationState) -> str:
        state.last_ownership_correction = "assistant_owned"
        if state.pending_assistant_question:
            return f"Да, это я спросила: {state.pending_assistant_question.rstrip('?')}?"
        question = extract_last_question(state.last_assistant_final)
        if question:
            state.pending_assistant_question = question
            return f"Да, это я спросила: {question.rstrip('?')}?"
        if state.last_assistant_final:
            return f"Да, это сказала я: {_quote(state.last_assistant_final)}."
        return "Да, это была моя реплика, не твоя."

    def _assistant_memory_reply(self, state: ConversationState) -> str:
        if state.pending_assistant_question:
            return f"Я спросила: {_quote(state.pending_assistant_question)}."
        question = extract_last_question(state.last_assistant_final)
        if question:
            return f"Я спросила: {_quote(question)}."
        if state.last_assistant_final:
            return f"Я сказала: {_quote(state.last_assistant_final)}."
        return "Я еще не успела ничего законченно сказать."

    def _user_memory_reply(
        self, intent: ConversationIntent, state: ConversationState, *, current_user_text: str
    ) -> str:
        target = intent.slots.get("target", "last")
        turns = self._previous_user_turns(state, current_user_text)
        if target == "called_as":
            if state.user_called_avatar_as:
                return f"Ты называл меня {state.user_called_avatar_as[-1]}."
            return "Пока никак конкретно не называл."
        if target == "start":
            if turns:
                return f"Ты начал с: {_quote(turns[0])}."
            return "У меня пока нет первой твоей реплики в памяти."
        if target == "previous":
            if len(turns) >= 2:
                return f"До этого ты сказал: {_quote(turns[-2])}."
            return "До этого у меня нет отдельной твоей реплики в памяти."
        if turns:
            return f"Ты только что спросил: {_quote(turns[-1])}."
        return "Я пока не успела поймать предыдущий вопрос."

    @staticmethod
    def _previous_user_turns(state: ConversationState, current_user_text: str) -> list[str]:
        turns = [turn.text for turn in state.user_turns if turn.status == "final"]
        if not turns:
            turns = list(state.last_user_turns)
        if turns and current_user_text and _normalize(turns[-1]) == _normalize(current_user_text):
            turns = turns[:-1]
        return turns

    @staticmethod
    def _first_user_turn_excluding_current(state: ConversationState, current_text: str) -> str:
        turns = ResponsePlanner._previous_user_turns(state, current_text)
        return turns[0] if turns else ""

    @staticmethod
    def _name_reply(intent: ConversationIntent, state: ConversationState) -> str:
        action = intent.slots.get("action", "")
        name = intent.slots.get("name", "")
        if action == "propose_name" and name:
            return f"Можно, {name} звучит лучше."
        if action == "reject_name":
            rejected = (
                name
                if name and name != "__current__"
                else (
                    state.candidate_name
                    or state.last_discussed_name
                    or _effective_avatar_name(state)
                )
            )
            return f"Окей, {rejected} вычеркиваем."
        if action == "why_name":
            return "Честно, просто выбрала неудачно. Если не нравится, сменим."
        return ""


@dataclass(slots=True)
class ConversationBrain:
    state: ConversationState = field(default_factory=ConversationState)
    router: IntentRouterProtocol = field(default_factory=IntentRouter)
    reducer: StateReducerProtocol | None = None
    planner: ResponsePlannerProtocol | None = None
    composer: PromptComposerProtocol = field(default_factory=PromptComposer)

    def __post_init__(self) -> None:
        if self.reducer is None:
            self.reducer = StateReducer(self.router)
        if self.planner is None:
            self.planner = ResponsePlanner(self.router, self.composer)

    def record_user_turn(
        self,
        user_text: str,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        return self.reducer.apply_user_turn(
            user_text,
            self.state,
            turn_id=turn_id,
            generation_id=generation_id,
        )

    def record_assistant_final(
        self,
        assistant_text: str,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        return self.reducer.apply_assistant_final(
            assistant_text,
            self.state,
            turn_id=turn_id,
            generation_id=generation_id,
        )

    def record_assistant_interrupted(
        self,
        assistant_text: str,
        *,
        turn_id: int = 0,
        generation_id: int = 0,
    ) -> ConversationState:
        return self.reducer.apply_assistant_interrupted(
            assistant_text,
            self.state,
            turn_id=turn_id,
            generation_id=generation_id,
        )

    def plan_response(
        self, history: list[dict[str, str]], user_text: str
    ) -> ResponsePlan:
        return self.planner.plan(user_text, self.state, history)

    def build_messages(
        self, history: list[dict[str, str]], user_text: str
    ) -> list[dict[str, str]]:
        return self.composer.compose(history, self.state, user_text)

    def direct_response(self, user_text: str) -> str | None:
        plan = self.plan_response([], user_text)
        return plan.direct_text if plan.mode == "direct" else None

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self.record_user_turn(user_text)
        self.record_assistant_final(assistant_text)


def split_questions(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    questions = [match.strip() for match in re.findall(r"[^.!?…]*\?", normalized)]
    consumed = sum(len(question) for question in questions)
    tail = normalized[consumed:].strip()
    if tail and _looks_like_question(tail):
        questions.append(f"{tail}?")
    return questions


def extract_last_question(text: str) -> str:
    questions = split_questions(text)
    return questions[-1] if questions else ""


def _looks_like_question(text: str) -> bool:
    normalized = _normalize(text)
    return normalized.startswith(
        (
            "как ",
            "что ",
            "с чего ",
            "почему ",
            "зачем ",
            "когда ",
            "где ",
            "кто ",
            "какой ",
            "какая ",
            "какие ",
        )
    )


DialogueBrain = ConversationBrain


def build_messages(history: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    """Compatibility wrapper for warmups and tests that do not own a brain."""

    return ConversationBrain().build_messages(history, user_text)
