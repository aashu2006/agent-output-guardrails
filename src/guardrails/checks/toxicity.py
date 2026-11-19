from collections.abc import Callable
from typing import Any
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

# Exact label names only (compared case-insensitively). Never substring matching:
# "non-toxic" must never be read as "toxic".
# unitary/toxic-bert labels + common binary-classifier labels.
TOXIC_LABELS = {
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
    "label_1", "1", "true", "unsafe",
}
SAFE_LABELS = {"non-toxic", "nontoxic", "non_toxic", "label_0", "0", "false", "safe"}

# Chunking so nothing past the model's 512-token window is silently dropped.
# Small windows on purpose: in a 500-token window a short insult after ~470 tokens
# of polite text scored 0.25 (diluted); at 64 tokens it scores ~0.97.
MAX_CHUNK_TOKENS = 64
CHUNK_OVERLAP_TOKENS = 16
MAX_CHUNK_WORDS = 48      # no tokenizer available: word window
CHUNK_OVERLAP_WORDS = 12


def _extract_toxicity_score(result: Any) -> float:
    """Extract toxicity probability score (0.0 to 1.0) for ONE text from various classifier outputs.
    Supports:
    - float/int directly
    - multi-label pipelines (unitary/toxic-bert): [{'label': 'toxic', 'score': 0.75},
      {'label': 'threat', 'score': 0.95}, ...] -> max over all toxic labels (0.95)
    - binary pipelines: [{'label': 'LABEL_1', 'score': 0.95}] or a safe label only,
      which gets inverted (1.0 - score)
    - nested [[{...}, ...]] and a single dict {'label': ..., 'score': ...}
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

        toxic_scores = []
        safe_score = None
        for item in result:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().lower()
            if label in TOXIC_LABELS:
                toxic_scores.append(float(item.get("score", 0.0)))
            elif label in SAFE_LABELS and safe_score is None:
                safe_score = float(item.get("score", 0.0))

        if toxic_scores:
            return max(toxic_scores)
        if safe_score is not None:
            return 1.0 - safe_score

    raise ValueError("classifier output has no recognised toxicity label")


def _batch_scores(raw_output: Any, n_texts: int) -> list[float]:
    """Per-text scores from a batch classifier call.
    Batch shape = one entry per text ([[labels...], [labels...]] or [0.1, 0.9]).
    Anything else (a scalar, a dict, one flat label list) is one result for the whole batch."""
    if (
        isinstance(raw_output, list)
        and raw_output
        and all(isinstance(x, (list, int, float)) for x in raw_output)
    ):
        if len(raw_output) != n_texts:
            raise ValueError(
                f"classifier returned {len(raw_output)} results for {n_texts} texts"
            )
        return [_extract_toxicity_score(x) for x in raw_output]
    return [_extract_toxicity_score(raw_output)]


def _collect_strings(obj: Any) -> list[str]:
    """String values from parsed JSON, depth-first. Keys, numbers, bools are skipped."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _collect_strings(v)]
    if isinstance(obj, list):
        return [s for item in obj for s in _collect_strings(item)]
    return []


def _word_chunks(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_CHUNK_WORDS:
        return [text]
    step = MAX_CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + MAX_CHUNK_WORDS]))
        if start + MAX_CHUNK_WORDS >= len(words):
            break
    return chunks


def _token_chunks(text: str, tokenizer: Any) -> list[str]:
    """Overlapping windows of MAX_CHUNK_TOKENS, cut from the original text via offsets."""
    offsets = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    if len(offsets) <= MAX_CHUNK_TOKENS:
        return [text]
    step = MAX_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    chunks = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + MAX_CHUNK_TOKENS]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + MAX_CHUNK_TOKENS >= len(offsets):
            break
    return chunks


class ToxicityScanner(Guardrail):
    name = "toxicity"

    def __init__(
        self,
        threshold: float = 0.8,
        model_name: str = "unitary/toxic-bert",
        classifier: Callable[[Any], Any] | None = None,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._classifier = classifier

    def _get_classifier(self) -> Callable[[Any], Any]:
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
                    batch_size=16,  # chunks go through the model together, not one by one
                )
            except ImportError as e:
                raise ImportError(
                    "transformers is required for ToxicityScanner default model. "
                    "Install transformers and torch, or pass a custom classifier callable."
                ) from e
        return self._classifier

    def _extract_score(self, raw_output: Any) -> float:
        return _extract_toxicity_score(raw_output)

    def _texts_to_score(self, candidate: str, context: dict | None) -> list[str]:
        """Each parsed JSON string value separately (keys/brackets never scored).
        No usable strings -> the raw candidate."""
        parsed = (context or {}).get("parsed")
        texts = []
        if parsed is not None:
            texts = [s for s in _collect_strings(parsed) if s.strip()]
        if not texts and candidate and candidate.strip():
            texts = [candidate]
        return texts

    def _chunk(self, text: str, classifier: Any) -> list[str]:
        tokenizer = getattr(classifier, "tokenizer", None)
        if tokenizer is not None:
            try:
                return _token_chunks(text, tokenizer)
            except Exception:
                pass  # slow tokenizer without offsets etc. -> word fallback
        return _word_chunks(text)

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        texts = self._texts_to_score(candidate, context)

        # 1. Nothing to score
        if not texts:
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=0.0,
                metadata={"score": 0.0, "threshold": self.threshold},
            )

        # 2. Har string/chunk ko ek batch call mein score karo, max = toxicity score
        classifier = self._get_classifier()
        chunks = [c for t in texts for c in self._chunk(t, classifier)]
        raw_output = classifier(chunks)
        score = max(_batch_scores(raw_output, len(chunks)))

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
