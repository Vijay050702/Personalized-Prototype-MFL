from __future__ import annotations

import time
from typing import Any

from app.core.logging import logger
from app.federated.models import AggregationRound


class RoundScheduler:
    def __init__(
        self,
        timeout_seconds: float = 300.0,
        min_clients: int = 1,
        allow_partial: bool = True,
        max_late_clients: int = 0,
    ):
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        if min_clients < 1:
            raise ValueError(f"min_clients must be >= 1, got {min_clients}")
        if max_late_clients < 0:
            raise ValueError(f"max_late_clients must be >= 0, got {max_late_clients}")

        self._timeout = timeout_seconds
        self._min_clients = min_clients
        self._allow_partial = allow_partial
        self._max_late_clients = max_late_clients

        self._current_round: AggregationRound | None = None
        self._round_counter: int = 0
        self._late_clients: set[str] = set()
        self._expected_clients: set[str] = set()

    @property
    def current_round_id(self) -> int:
        return self._round_counter

    @property
    def current_round(self) -> AggregationRound | None:
        return self._current_round

    def new_round(self, client_ids: list[str]) -> AggregationRound:
        self._round_counter += 1
        self._expected_clients = set(client_ids)
        self._late_clients.clear()

        round_obj = AggregationRound(
            round_id=self._round_counter,
            participating_clients=list(client_ids),
            status="active",
        )
        self._current_round = round_obj
        logger.info(
            f"Started round {self._round_counter} with {len(client_ids)} clients"
        )
        return round_obj

    def client_arrived(self, client_id: str) -> bool:
        if self._current_round is None:
            return False
        if client_id in self._expected_clients:
            if client_id not in self._current_round.participating_clients:
                self._current_round.participating_clients.append(client_id)
            return True
        else:
            if len(self._late_clients) < self._max_late_clients:
                self._late_clients.add(client_id)
                self._current_round.participating_clients.append(client_id)
                logger.info(f"Late client {client_id} accepted")
                return True
            else:
                logger.warning(f"Late client {client_id} rejected")
                return False

    def has_timed_out(self) -> bool:
        if self._current_round is None:
            return False
        elapsed = time.time() - self._current_round.started_at
        return elapsed > self._timeout

    def time_remaining(self) -> float:
        if self._current_round is None:
            return 0.0
        elapsed = time.time() - self._current_round.started_at
        return max(0.0, self._timeout - elapsed)

    def can_finalize(self) -> bool:
        if self._current_round is None:
            return False
        actual = len(self._current_round.participating_clients)
        expected = len(self._expected_clients)
        if actual >= expected:
            return True
        if self._allow_partial and actual >= self._min_clients:
            return True
        if self.has_timed_out() and actual >= self._min_clients:
            return True
        return False

    def finalize_round(self) -> AggregationRound:
        if self._current_round is None:
            raise ValueError("No active round to finalize")
        self._current_round.complete()
        round_obj = self._current_round
        logger.info(
            f"Finalized round {round_obj.round_id} with "
            f"{len(round_obj.participating_clients)} clients"
        )
        self._current_round = None
        return round_obj

    def abort_round(self) -> None:
        if self._current_round is not None:
            self._current_round.status = "aborted"
            logger.warning(f"Aborted round {self._current_round.round_id}")
            self._current_round = None

    def to_config(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self._timeout,
            "min_clients": self._min_clients,
            "allow_partial": self._allow_partial,
            "max_late_clients": self._max_late_clients,
        }
