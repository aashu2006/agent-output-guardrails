from abc import ABC, abstractmethod
from guardrails.results import GuardrailResult


class Guardrail(ABC):
    name: str

    def check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        try:
            return self._check(candidate, context)
        except Exception as e:
            return GuardrailResult(
                guardrail=self.name, passed=False, error=True,
                reason=f"internal error: {type(e).__name__}",
            )

    @abstractmethod
    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        ...