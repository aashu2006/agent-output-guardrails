from collections.abc import Callable
from typing import Any
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

# Exact label names only (compared case-insensitively). Never substring matching:
# "non-toxic" must never be read as "toxic".
TOXIC_LABELS = {"toxic", "label_1", "1", "true", "unsafe"}
SAFE_LABELS = {"non-toxic", "nontoxic", "non_toxic", "label_0", "0", "false", "safe"}


def _extract_toxicity_score(result: Any) -> float:
    """Extract toxicity probability score (0.0 to 1.0) from various classifier outputs.
    Supports:
    - float/int directly
    - multi-label pipelines (unitary/toxic-bert): [{'label': 'toxic', 'score': 0.95},
      {'label': 'insult', 'score': 0.9}, ...] -> score of the exact "toxic" label
    - binary pipelines: [{'label': 'LABEL_1', 'score': 0.95}] or a safe label only,
      which gets inverted (1.0 - score)
    - batch-shaped [[{...}, ...]] and a single dict {'label': ..., 'score': ...}
    Output with labels we don't recognise raises ValueError, so the base check()
    turns it into an error result and policy fails closed.
    """
    if isinstance(result, (float, int)):
        return float(result)

    if isinstance(result, dict):
        result = [result]

    if isinstance(result, list):
        if len(result) == 0:
            return 0.0
        # Pipeline might return [[{...}, {...}]] for batch or [{...}, {...}] for single
        if isinstance(result[0], list):
            result = result[0]

        toxic_score = None
        safe_score = None
        for item in result:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().lower()
            if label in TOXIC_LABELS and toxic_score is None:
                toxic_score = float(item.get("score", 0.0))
            elif label in SAFE_LABELS and safe_score is None:
                safe_score = float(item.get("score", 0.0))

        if toxic_score is not None:
            return toxic_score
        if safe_score is not None:
            return 1.0 - safe_score

    raise ValueError("classifier output has no recognised toxicity label")


def _collect_strings(obj: Any) -> list[str]:
    """String values from parsed JSON, depth-first. Keys, numbers, bools are skipped."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _collect_strings(v)]
    if isinstance(obj, list):
        return [s for item in obj for s in _collect_strings(item)]
    return []


class ToxicityScanner(Guardrail):
    name = "toxicity"

    def __init__(
        self,
        threshold: float = 0.8,
        model_name: str = "unitary/toxic-bert",
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
                    truncation=True,
                    max_length=512,
                    function_to_apply="sigmoid",  # multi-label: independent per-label probs
                )
            except ImportError as e:
                raise ImportError(
                    "transformers is required for ToxicityScanner default model. "
                    "Install transformers and torch, or pass a custom classifier callable."
                ) from e
        return self._classifier

    def _extract_score(self, raw_output: Any) -> float:
        return _extract_toxicity_score(raw_output)

    def _text_to_score(self, candidate: str, context: dict | None) -> str:
        """Parsed JSON present -> only its string values, so keys/brackets don't skew the score."""
        parsed = (context or {}).get("parsed")
        if parsed is not None:
            return "\n".join(_collect_strings(parsed))
        return candidate

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        text = self._text_to_score(candidate, context)

        # 1. Empty string case
        if not text or not text.strip():
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=0.0,
                metadata={"score": 0.0, "threshold": self.threshold},
            )

        # 2. Classifier run karo aur toxicity score nikalo (0.0 to 1.0)
        classifier = self._get_classifier()
        raw_output = classifier(text)
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
