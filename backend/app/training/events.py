from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class EventType(Enum):
    EXPERIMENT_START = auto()
    EXPERIMENT_END = auto()
    ROUND_START = auto()
    ROUND_END = auto()
    CLIENT_SELECTED = auto()
    CLIENT_TRAINING_START = auto()
    CLIENT_TRAINING_END = auto()
    CLIENT_UPLOAD = auto()
    AGGREGATION_START = auto()
    AGGREGATION_END = auto()
    KNOWLEDGE_TRANSFER_START = auto()
    KNOWLEDGE_TRANSFER_END = auto()
    PERSONALIZATION_START = auto()
    PERSONALIZATION_END = auto()
    SYNCHRONIZATION_START = auto()
    SYNCHRONIZATION_END = auto()
    EVALUATION_START = auto()
    EVALUATION_END = auto()
    CHECKPOINT_SAVED = auto()
    CHECKPOINT_LOADED = auto()
    ERROR_OCCURRED = auto()
    EARLY_STOP = auto()


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


EventHandler = Callable[[Event], None]


class EventDispatcher:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}

    def register(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unregister(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    def dispatch(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            handler(event)

    def dispatch_simple(
        self, event_type: EventType, data: dict[str, Any] | None = None
    ) -> None:
        self.dispatch(Event(type=event_type, data=data or {}))

    def clear(self) -> None:
        self._handlers.clear()

    @property
    def handler_count(self) -> dict[EventType, int]:
        return {et: len(hs) for et, hs in self._handlers.items()}
