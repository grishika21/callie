"""A scripted chat model for agent tests (the pattern of tests/fakes.py, for LangChain).

Each model call takes the next step of the script: an AIMessage, a callable that builds
one from the messages, or the string "sleep:<seconds>". Running out of steps fails the
test, so a test also proves how many model calls a turn made.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = itertools.count(1)


def call(name: str, **args: Any) -> AIMessage:
    return calls((name, args))


def calls(*pairs: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"call_{next(_ids)}", "type": "tool_call"}
                                     for n, a in pairs])


def say(text: str) -> AIMessage:
    return AIMessage(text)


class ScriptedChatModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)  # the messages of every call
    bound: list[list[str]] = Field(default_factory=list)  # tool names offered to the model

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self

    def _next(self, messages: list[BaseMessage]) -> Any:
        self.seen.append(list(messages))
        if not self.script:
            raise AssertionError(f"unexpected model call #{len(self.seen)}")
        step = self.script.pop(0)
        return step(messages) if isinstance(step, Callable) else step  # type: ignore[arg-type]

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        step = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=step.model_copy())])

    async def _agenerate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                         **kwargs: Any) -> ChatResult:
        step = self._next(messages)
        if isinstance(step, str) and step.startswith("sleep:"):
            await asyncio.sleep(float(step.split(":")[1]))
            step = say("too late")
        return ChatResult(generations=[ChatGeneration(message=step.model_copy())])
