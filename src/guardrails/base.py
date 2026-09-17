from abc import ABC, abstractmethod
from guardrails.results import GuardrailResult


class Guardrail(ABC):
    name: str

    @abstractmethod
    def check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        ...