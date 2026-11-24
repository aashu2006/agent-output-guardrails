from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract contract for LLM inference providers."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generates a completion string from the model given a prompt."""
        ...


class FakeProvider(LLMProvider):
    """Deterministic provider for testing and reproducible unit tests.
    Returns scripted responses in FIFO order and records prompt history."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.history: list[str] = []
        self.call_count: int = 0

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.history.append(prompt)
        self.call_count += 1
        if not self.responses:
            raise RuntimeError(
                f"FakeProvider ran out of scripted responses after {self.call_count - 1} calls"
            )
        return self.responses.pop(0)
