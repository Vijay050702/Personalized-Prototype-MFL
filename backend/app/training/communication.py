from __future__ import annotations

import hashlib
import json
import time
import zlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    sender_id: str
    receiver_id: str
    round_id: int
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    timestamp: float = field(default_factory=time.time)
    compression: str = "none"
    original_size: int = 0
    compressed_size: int = 0

    def verify_checksum(self) -> bool:
        payload_str = json.dumps(self.payload, sort_keys=True)
        expected = hashlib.sha256(payload_str.encode()).hexdigest()
        return self.checksum == expected


class CommunicationLayer:
    def __init__(self) -> None:
        self._messages_sent: list[Message] = []
        self._messages_received: list[Message] = []
        self._latencies: list[float] = []
        self._compression_enabled: bool = True
        self._checksum_enabled: bool = True

    @property
    def compression_enabled(self) -> bool:
        return self._compression_enabled

    def set_compression(self, enabled: bool) -> None:
        self._compression_enabled = enabled

    @property
    def checksum_enabled(self) -> bool:
        return self._checksum_enabled

    def set_checksum(self, enabled: bool) -> None:
        self._checksum_enabled = enabled

    def send(
        self,
        sender_id: str,
        receiver_id: str,
        round_id: int,
        message_type: str,
        payload: dict[str, Any],
    ) -> Message:
        payload_bytes = json.dumps(payload).encode()
        original_size = len(payload_bytes)

        compressed = payload_bytes
        compression = "none"
        if self._compression_enabled and len(payload_bytes) > 1024:
            compressed = zlib.compress(payload_bytes)
            compression = "zlib"

        checksum = ""
        if self._checksum_enabled:
            checksum = hashlib.sha256(payload_bytes).hexdigest()

        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            round_id=round_id,
            message_type=message_type,
            payload=payload,
            checksum=checksum,
            compression=compression,
            original_size=original_size,
            compressed_size=len(compressed),
        )
        self._messages_sent.append(message)
        return message

    def receive(
        self,
        message: Message,
    ) -> dict[str, Any]:
        start = time.time()

        if self._checksum_enabled and not message.verify_checksum():
            raise ValueError(f"Checksum mismatch for message from {message.sender_id}")

        self._messages_received.append(message)
        latency = time.time() - start
        self._latencies.append(latency)
        return message.payload

    def send_batch(
        self,
        sender_id: str,
        receiver_id: str,
        round_id: int,
        messages: list[tuple[str, dict[str, Any]]],
    ) -> list[Message]:
        sent: list[Message] = []
        for msg_type, payload in messages:
            msg = self.send(sender_id, receiver_id, round_id, msg_type, payload)
            sent.append(msg)
        return sent

    def receive_batch(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        return [self.receive(msg) for msg in messages]

    @property
    def messages_sent_count(self) -> int:
        return len(self._messages_sent)

    @property
    def messages_received_count(self) -> int:
        return len(self._messages_received)

    @property
    def average_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def max_latency(self) -> float:
        return max(self._latencies) if self._latencies else 0.0

    @property
    def total_bytes_sent(self) -> int:
        return sum(m.original_size for m in self._messages_sent)

    @property
    def total_bytes_compressed(self) -> int:
        return sum(m.compressed_size for m in self._messages_sent)

    def payload_statistics(self) -> dict[str, Any]:
        if not self._messages_sent:
            return {
                "total_messages": 0,
                "total_bytes": 0,
                "avg_bytes": 0.0,
                "compression_ratio": 1.0,
            }
        total_orig = self.total_bytes_sent
        total_comp = self.total_bytes_compressed
        return {
            "total_messages": self.messages_sent_count,
            "total_bytes": total_orig,
            "avg_bytes": total_orig / self.messages_sent_count,
            "compression_ratio": total_comp / max(total_orig, 1),
        }

    def clear(self) -> None:
        self._messages_sent.clear()
        self._messages_received.clear()
        self._latencies.clear()
