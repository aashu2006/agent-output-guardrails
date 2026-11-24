import re
import math
from collections.abc import Callable
from typing import Any
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

# Defaults from system architecture plan
DEFAULT_T_LOW = 0.35      # Below this: skip NLI, mark ungrounded with score 0.0
DEFAULT_T_GROUND = 0.70   # NLI entailment prob >= T_ground for claim to pass
DEFAULT_CHUNK_SIZE = 200  # Words/tokens per source chunk
DEFAULT_CHUNK_OVERLAP = 40
MIN_CLAIM_WORDS = 4       # Skip fragments < 4 words

# Common greetings and conversational meta-phrases to filter out
GREETING_PATTERNS = [
    re.compile(r"^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening|day))\b", re.I),
    re.compile(r"\b(thanks|thank\s+you|you('re|\s+are)\s+welcome)\b", re.I),
    re.compile(r"\b(have\s+a\s+(great|nice|wonderful|good)\s+day)\b", re.I),
    re.compile(r"^(sure|certainly|of\s+course)[\s,!.:-]*(here\s+(is|are).*)?$", re.I),
    re.compile(r"^here\s+(is|are)(\s+the)?(\s+information|\s+answer|\s+response)?[\s,!.:-]*$", re.I),
    re.compile(r"^(based\s+on\s+the\s+(provided\s+)?(context|text|document|source))", re.I),
]

# Titles that are never sentence endings
HONORIFICS = r"Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Sr\.|Jr\."
# Non-terminal abbreviations
COMMON_ABBR = r"e\.g\.|i\.e\.|vs\.|etc\.|Inc\.|Ltd\.|Co\."


def _collect_strings(obj: Any) -> list[str]:
    """Extracts string values from parsed JSON tree, depth-first. Keys/numbers/booleans are skipped."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _collect_strings(v)]
    if isinstance(obj, list):
        return [s for item in obj for s in _collect_strings(item)]
    return []


def _split_sentences(text: str) -> list[str]:
    """Splits text into clean sentences, preserving abbreviations (Dr., e.g.) and numbers (3.14)."""
    if not text or not text.strip():
        return []

    # Protect decimals: replace e.g. 3.14 with 3<DOT>14
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text.strip())

    # Protect titles and common abbreviations
    def _protect(match: re.Match) -> str:
        return match.group(0).replace(".", "<DOT>")

    protected = re.sub(rf"\b({HONORIFICS})\s+", _protect, protected, flags=re.I)
    protected = re.sub(rf"\b({COMMON_ABBR})\s+", _protect, protected, flags=re.I)

    # Protect single initials like 'U.' in 'U.S.' when followed by another initial
    protected = re.sub(r"\b([A-Z])\.(?=[A-Z]\.)", r"\1<DOT>", protected)

    # Split on terminal punctuation followed by space and capital letter, quote, or end of string
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])", protected)

    sentences = []
    for s in raw_sentences:
        restored = s.replace("<DOT>", ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


def _is_claim(sentence: str, min_words: int = MIN_CLAIM_WORDS) -> bool:
    """Claim filter: skips questions, greetings, meta-talk, and short fragments (< min_words)."""
    clean = sentence.strip()
    words = clean.split()
    if len(words) < min_words:
        return False

    # Skip questions
    if clean.endswith("?"):
        return False
    if re.match(r"^(who|what|where|when|why|how|is|are|can|could|would|should|do|does|did)\b", clean, re.I) and "?" in clean:
        return False

    # Skip greetings and meta chatter
    for pattern in GREETING_PATTERNS:
        if pattern.search(clean):
            return False

    return True


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Splits source document into overlapping chunks."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text.strip()]

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def _default_lexical_similarity(text1: str, text2: str) -> float:
    """Fast fallback lexical/token cosine similarity if no vector embedder is injected."""
    def _term_vec(s: str) -> dict[str, int]:
        vec: dict[str, int] = {}
        for w in re.findall(r"\w+", s.lower()):
            vec[w] = vec.get(w, 0) + 1
        return vec

    v1, v2 = _term_vec(text1), _term_vec(text2)
    if not v1 or not v2:
        return 0.0

    common = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class GroundednessChecker(Guardrail):
    name = "groundedness"

    def __init__(
        self,
        t_low: float = DEFAULT_T_LOW,
        t_ground: float = DEFAULT_T_GROUND,
        top_k: int = 3,
        max_source_chunks: int = 50,
        embedder: Callable[[str, list[str]], list[float]] | None = None,
        nli_classifier: Callable[[str, str], float] | None = None,
        min_words: int = MIN_CLAIM_WORDS,
    ):
        """
        Groundedness checker adhering to the 'Cosine only rejects, NLI decides' architecture.
        - t_low: Minimum cosine similarity to avoid early rejection (score 0.0).
        - t_ground: Minimum NLI entailment probability required for claim to be considered grounded.
        - top_k: Number of highest-similarity chunks evaluated per claim.
        - max_source_chunks: Maximum source chunks to evaluate to prevent latency blowup.
        - embedder: Optional custom (query, docs) -> similarities callable.
        - nli_classifier: Optional custom (premise, hypothesis) -> entailment_prob callable.
        """
        self.t_low = t_low
        self.t_ground = t_ground
        self.top_k = max(1, top_k)
        self.max_source_chunks = max(1, max_source_chunks)
        self.min_words = min_words
        self._embedder = embedder
        self._nli_classifier = nli_classifier

    def _get_embedder(self) -> Callable[[str, list[str]], list[float]]:
        if self._embedder is not None:
            return self._embedder

        # Try sentence-transformers bi-encoder if available, else lexical cosine
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")

            def _st_embedder(query: str, docs: list[str]) -> list[float]:
                q_emb = model.encode(query, convert_to_tensor=True)
                d_emb = model.encode(docs, convert_to_tensor=True)
                sims = util.cos_sim(q_emb, d_emb)[0].tolist()
                return [float(s) for s in sims]

            self._embedder = _st_embedder
            return self._embedder
        except Exception:
            # Fallback to zero-dependency term-cosine
            def _fallback_embedder(query: str, docs: list[str]) -> list[float]:
                return [_default_lexical_similarity(query, d) for d in docs]

            self._embedder = _fallback_embedder
            return self._embedder

    def _get_nli(self) -> Callable[[str, str], float]:
        if self._nli_classifier is not None:
            return self._nli_classifier

        try:
            from transformers import pipeline
            nli_pipe = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-xsmall",
                top_k=None,
                truncation=True,
            )

            def _hf_nli(premise: str, hypothesis: str) -> float:
                # Proper HuggingFace cross-encoder text pair format
                try:
                    res = nli_pipe({"text": premise, "text_pair": hypothesis})
                except Exception:
                    res = nli_pipe(f"{premise} [SEP] {hypothesis}")

                if isinstance(res, list) and res:
                    items = res[0] if isinstance(res[0], list) else res
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        lbl = str(item.get("label", "")).lower()
                        if "entail" in lbl:
                            return float(item.get("score", 0.0))
                    # Fallback if no explicit entailment label
                    top_item = items[0]
                    lbl = str(top_item.get("label", "")).lower()
                    score = float(top_item.get("score", 0.0))
                    if "contradict" in lbl:
                        return 0.0
                    return 1.0 - score
                return 0.0

            self._nli_classifier = _hf_nli
            return self._nli_classifier
        except Exception as e:
            raise ImportError(
                "NLI classifier requires transformers and torch or a mock callable."
            ) from e

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        context = context or {}
        source = context.get("source")

        # Note 2: No source given -> skip the check, passed=True, note it in metadata
        if not source or not str(source).strip():
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=1.0,
                metadata={"skipped": "no_source", "reason": "no source context provided"},
            )

        source_str = str(source).strip()

        # Step 1: Extract text values safely (if structured JSON parsed tree present)
        parsed = context.get("parsed")
        raw_texts: list[str] = []
        if parsed is not None:
            raw_texts = [s for s in _collect_strings(parsed) if s.strip()]
        if not raw_texts and candidate and candidate.strip():
            raw_texts = [candidate]

        sentences: list[str] = []
        for txt in raw_texts:
            sentences.extend(_split_sentences(txt))

        if not sentences:
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=1.0,
                metadata={"claims_evaluated": 0},
            )

        # Step 2: Claim filter (skip questions, greetings, meta, < min_words)
        claims = [s for s in sentences if _is_claim(s, self.min_words)]
        if not claims:
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=1.0,
                metadata={"claims_evaluated": 0, "skipped_sentences": sentences},
            )

        # Step 3: Chunk + embed source
        chunks = _chunk_text(source_str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
        if len(chunks) > self.max_source_chunks:
            chunks = chunks[:self.max_source_chunks]
        embed_fn = self._get_embedder()
        nli_fn = None  # Lazy initialize NLI only if a claim needs it

        claim_scores: dict[str, float] = {}
        claim_details: list[dict[str, Any]] = []

        for claim in claims:
            similarities = embed_fn(claim, chunks)
            if similarities:
                # Rank chunks by similarity descending
                ranked_chunks = sorted(
                    enumerate(similarities), key=lambda x: x[1], reverse=True
                )
                top_candidates = ranked_chunks[:self.top_k]
                best_sim = top_candidates[0][1]
            else:
                top_candidates = [(0, 0.0)]
                best_sim = 0.0

            # Step 4: Cosine >= T_low?
            if best_sim < self.t_low:
                # UNGROUNDED: no related source, skip NLI, score = 0.0
                claim_score = 0.0
                claim_scores[claim] = claim_score
                claim_details.append({
                    "claim": claim,
                    "best_sim": round(best_sim, 4),
                    "action": "rejected_by_cosine",
                    "score": claim_score,
                })
            else:
                # Step 5: NLI (DeBERTa) decides every real claim
                if nli_fn is None:
                    nli_fn = self._get_nli()

                # Evaluate NLI against top candidate chunks that pass t_low, take max entailment
                candidate_nli_scores = []
                for idx, sim in top_candidates:
                    if sim < self.t_low:
                        continue
                    chunk_text = chunks[idx] if idx < len(chunks) else source_str
                    candidate_nli_scores.append(float(nli_fn(chunk_text, claim)))

                claim_score = max(candidate_nli_scores) if candidate_nli_scores else 0.0
                claim_scores[claim] = claim_score
                claim_details.append({
                    "claim": claim,
                    "best_sim": round(best_sim, 4),
                    "action": "evaluated_by_nli",
                    "score": round(claim_score, 4),
                    "chunks_evaluated": len(candidate_nli_scores),
                })

        # Step 6: score = min(claims)
        min_score = min(claim_scores.values()) if claim_scores else 1.0
        ungrounded = [c for c, sc in claim_scores.items() if sc < self.t_ground]
        passed = min_score >= self.t_ground

        reason = (
            None
            if passed
            else f"groundedness score {min_score:.2f} below threshold {self.t_ground:.2f}: {len(ungrounded)} ungrounded claim(s)"
        )

        return GuardrailResult(
            guardrail=self.name,
            passed=passed,
            score=round(min_score, 4),
            reason=reason,
            metadata={
                "score": round(min_score, 4),
                "threshold": self.t_ground,
                "t_low": self.t_low,
                "claims_evaluated": len(claims),
                "ungrounded": ungrounded,
                "details": claim_details,
            },
        )
