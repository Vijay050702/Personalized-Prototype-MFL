from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    round_id: int = 0
    client_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error: Exception | None = None


class Hook(ABC):
    @abstractmethod
    def execute(self, context: HookContext) -> None:
        pass


class HookManager:
    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = {
            "before_round": [],
            "after_round": [],
            "before_training": [],
            "after_training": [],
            "before_aggregation": [],
            "after_aggregation": [],
            "before_knowledge_transfer": [],
            "after_knowledge_transfer": [],
            "before_personalization": [],
            "after_personalization": [],
            "before_synchronization": [],
            "after_synchronization": [],
            "before_evaluation": [],
            "after_evaluation": [],
            "before_checkpoint": [],
            "after_checkpoint": [],
            "on_error": [],
        }

    def register(self, hook_point: str, hook: Hook) -> None:
        if hook_point not in self._hooks:
            raise ValueError(
                f"Unknown hook point '{hook_point}'. "
                f"Available: {list(self._hooks.keys())}"
            )
        self._hooks[hook_point].append(hook)

    def unregister(self, hook_point: str, hook: Hook) -> None:
        if hook_point in self._hooks:
            self._hooks[hook_point] = [
                h for h in self._hooks[hook_point] if h is not hook
            ]

    def execute(self, hook_point: str, context: HookContext) -> None:
        for hook in self._hooks.get(hook_point, []):
            try:
                hook.execute(context)
            except Exception as e:
                context.error = e

    def available_hook_points(self) -> list[str]:
        return list(self._hooks.keys())

    def hook_count(self, hook_point: str) -> int:
        return len(self._hooks.get(hook_point, []))

    def total_hooks(self) -> int:
        return sum(len(hooks) for hooks in self._hooks.values())

    def clear(self) -> None:
        for hook_point in self._hooks:
            self._hooks[hook_point].clear()
