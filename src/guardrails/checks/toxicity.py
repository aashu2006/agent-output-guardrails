from collections.abc import Callable
from typing import Any
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult


def _extract_toxicity_score(result: Any) -> float:
    """Extract toxicity probability score (0.0 to 1.0) from various classifier outputs.
    Supports:
    - float/int directly
    - HuggingFace pipelines returning [{'label': 'toxic', 'score': 0.95}, ...]
    - HuggingFace pipelines returning [{'label': 'LABEL_1', 'score': 0.95}]
    - Single dict {'label': ..., 'score': ...}
    """
    if isinstance(result, (float, int)):
        return float(result)

    if isinstance(result, list):
        if len(result) == 0:
            return 0.0
        # Pipeline might return [[{...}, {...}]] for batch or [{...}, {...}] for single
        if isinstance(result[0], list):
            result = result[0]

        # First check for non-toxic / safe labels
        non_toxic_score = None
        for item in result:
            if isinstance(item, dict):
                label = str(item.get("label", "")).lower()
                if "non-toxic" in label or "nontoxic" in label or label in ("label_0", "0", "false", "safe"):
                    non_toxic_score = 1.0 - float(item.get("score", 0.0))
                elif "toxic" in label or label in ("label_1", "1", "true", "unsafe"):
                    return float(item.get("score", 0.0))

        if non_toxic_score is not None:
            return non_toxic_score

        # Fallback to single item score
        if len(result) == 1 and isinstance(result[0], dict):
            return float(result[0].get("score", 0.0))

    if isinstance(result, dict):
        label = str(result.get("label", "")).lower()
        score = float(result.get("score", 0.0))
        if "non-toxic" in label or "nontoxic" in label or label in ("label_0", "safe"):
            return 1.0 - score
        return score

    return 0.0


class ToxicityScanner(Guardrail):
    name = "toxicity"

    def __init__(
        self,
        threshold: float = 0.8,
        model_name: str = "martin-ha/toxic-comment-model",
        classifier: Callable[[str], Any] | None = None,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._classifier = classifier

    def _get_classifier(self) -> Callable[[str], Any]:
        """Lazy load HuggingFace pipeline taaki import time fast rahe."""
        if self._classifier is None:
            try:
                from transformers import pipeline
                self._classifier = pipeline(
                    "text-classification",
                    model=self.model_name,
                    top_k=None,
                )
            except ImportError as e:
                raise ImportError(
                    "transformers is required for ToxicityScanner default model. "
                    "Install transformers and torch, or pass a custom classifier callable."
                ) from e
        return self._classifier

    def _extract_score(self, raw_output: Any) -> float:
        return _extract_toxicity_score(raw_output)

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        # 1. Empty string case
        if not candidate or not candidate.strip():
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=0.0,
                metadata={"score": 0.0, "threshold": self.threshold},
            )

        # 2. Classifier run karo aur toxicity score nikalo (0.0 to 1.0)
        classifier = self._get_classifier()
        raw_output = classifier(candidate)
        score = self._extract_score(raw_output)

        # 3. Check pass/fail
        passed = score < self.threshold
        reason = (
            None
            if passed
            else f"toxicity score {score:.2f} exceeds threshold {self.threshold:.2f}"
        )

        return GuardrailResult(
            guardrail=self.name,
            passed=passed,
            score=round(score, 4),
            reason=reason,
            metadata={"score": round(score, 4), "threshold": self.threshold},
        )
