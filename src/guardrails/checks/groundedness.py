import re
from collections.abc import Callable
from typing import Any, Literal
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

# Defaults from system architecture plan
DEFAULT_T_LOW = 0.35      # Below this: skip NLI, mark ungrounded with score 0.0
DEFAULT_T_GROUND = 0.70   # NLI entailment prob >= T_ground for claim to pass
DEFAULT_CHUNK_SIZE = 200  # Words/tokens per source chunk
DEFAULT_CHUNK_OVERLAP = 40
MIN_CLAIM_WORDS = 4       # Skip fragments < 4 words

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

# Batched interfaces:
#   embedder(claims, chunks) -> similarity matrix, shape [len(claims)][len(chunks)]
#   nli_classifier([(premise, hypothesis), ...]) -> entailment probability per pair
Embedder = Callable[[list[str], list[str]], list[list[float]]]
NLIClassifier = Callable[[list[tuple[str, str]]], list[float]]

_PUNCT = r"[\s]*[,!.:;\-–—]+\s*"

# Leading filler that is stripped before the rest of the sentence is checked as a claim.
# Anchored at the start; each alternative must end in punctuation so real content is never eaten.
FILLER_PREFIX = re.compile(
    r"^(?:"
    r"(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day))(?:\s+there)?" + _PUNCT + r"|"
    r"(?:sure|certainly|of\s+course|absolutely|okay|ok|great)" + _PUNCT + r"|"
    r"(?:sure|certainly|of\s+course|absolutely)\s+(?=here\s+(?:is|are)\b)|"
    r"here\s+(?:is|are)(?:\s+the)?"
    r"(?:\s+(?:information|answer|response|summary|details|result|results))?"
    r"(?:\s+you\s+(?:asked|requested)\s+for)?" + _PUNCT + r"|"
    r"(?:based\s+on|according\s+to)\s+the\s+(?:provided\s+|given\s+|available\s+)?"
    r"(?:context|text|document|documents|source|sources|information|data)" + _PUNCT +
    r")",
    re.I,
)

# Whole-sentence chatter: a sentence is skipped only if it fully matches one of these
# (after leading filler is stripped and trailing punctuation removed).
CHATTER_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day))(?:\s+\w+)?",
        r"(?:many\s+)?(?:thanks|thank\s+you)(?:\s+(?:so|very)\s+much)?(?:\s+for\s+.*)?",
        r"you(?:'re|\s+are)\s+(?:very\s+|most\s+)?welcome.*",
        r"(?:i\s+)?hope\s+(?:this|that|it|the\s+above)\s+(?:helps|is\s+helpful|answers\s+your\s+question).*",
        r"(?:please\s+)?(?:do\s+)?let\s+me\s+know\b.*",
        r"(?:please\s+)?(?:feel\s+free|don'?t\s+hesitate|do\s+not\s+hesitate)\s+to\b.*",
        r"if\s+you\s+(?:have|need)\s+(?:any\s+)?(?:other\s+|more\s+|further\s+|additional\s+)?"
        r"(?:questions|help|assistance|information|clarification).*",
        r"(?:i'?m|i\s+am)\s+(?:happy|glad)\s+to\s+help.*",
        r"(?:glad|happy)\s+(?:i\s+could|to)\s+help.*",
        r"have\s+a\s+(?:great|nice|wonderful|good|lovely)\s+(?:day|evening|week|weekend|one).*",
        r"(?:is\s+there\s+)?anything\s+else\s+i\s+can\s+(?:help|do).*",
    ]
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


def _extract_claim(sentence: str, min_words: int = MIN_CLAIM_WORDS) -> str | None:
    """Returns the checkable claim text, or None when the sentence is not a claim.

    Leading filler ("Sure, here is the answer:", "Based on the provided context,") is
    stripped and the REST is checked. A sentence is skipped only when it is a question,
    pure chatter/filler, or a fragment shorter than min_words."""
    text = sentence.strip()
    if not text or text.endswith("?"):
        return None

    previous = None
    while previous != text:
        previous = text
        text = FILLER_PREFIX.sub("", text, count=1).strip()

    core = text.rstrip(" .!")
    if not core:
        return None
    if any(p.fullmatch(core) for p in CHATTER_PATTERNS):
        return None
    if len(text.split()) < min_words:
        return None
    return text[0].upper() + text[1:]


def _is_claim(sentence: str, min_words: int = MIN_CLAIM_WORDS) -> bool:
    """Claim filter: skips questions, pure chatter/filler, and short fragments (< min_words)."""
    return _extract_claim(sentence, min_words) is not None


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


class GroundednessChecker(Guardrail):
    name = "groundedness"

    def __init__(
        self,
        t_low: float = DEFAULT_T_LOW,
        t_ground: float = DEFAULT_T_GROUND,
        top_k: int = 3,
        max_source_chunks: int | None = None,
        embedder: Embedder | None = None,
        nli_classifier: NLIClassifier | None = None,
        min_words: int = MIN_CLAIM_WORDS,
        on_missing_source: Literal["block", "skip"] = "block",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
    ):
        """
        Groundedness checker adhering to the 'Cosine only rejects, NLI decides' architecture.
        - t_low: Minimum cosine similarity to avoid early rejection (score 0.0).
        - t_ground: Minimum NLI entailment probability required for claim to be considered grounded.
        - top_k: Number of highest-similarity chunks sent to NLI per claim.
        - max_source_chunks: Optional cap on source chunks. None (default) ranks across ALL chunks;
          when a cap truncates the source, metadata records source_truncated=True.
        - embedder: Optional batched (claims, chunks) -> similarity-matrix callable.
        - nli_classifier: Optional batched [(premise, hypothesis)] -> entailment-probs callable.
        - on_missing_source: "block" (default) fails closed when no source is given; "skip" passes.
        """
        if on_missing_source not in ("block", "skip"):
            raise ValueError("on_missing_source must be 'block' or 'skip'")
        self.t_low = t_low
        self.t_ground = t_ground
        self.top_k = max(1, top_k)
        self.max_source_chunks = None if max_source_chunks is None else max(1, max_source_chunks)
        self.min_words = min_words
        self.on_missing_source = on_missing_source
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self._embedder = embedder
        self._nli_classifier = nli_classifier

    def _get_embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder

        # No silent fallback: if the bi-encoder can't load, raise -> crash-safe error result -> BLOCK
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as e:
            raise ImportError(
                "GroundednessChecker default embedder requires sentence-transformers "
                "(pip install -e '.[ml]') or an injected embedder callable."
            ) from e
        model = SentenceTransformer(self.embedding_model)

        def _st_embedder(claims: list[str], chunks: list[str]) -> list[list[float]]:
            # Source chunks encoded once per call, all claims batch-encoded in one call
            chunk_emb = model.encode(chunks, convert_to_tensor=True, batch_size=32)
            claim_emb = model.encode(claims, convert_to_tensor=True, batch_size=32)
            return util.cos_sim(claim_emb, chunk_emb).tolist()

        self._embedder = _st_embedder
        return self._embedder

    def _get_nli(self) -> NLIClassifier:
        if self._nli_classifier is not None:
            return self._nli_classifier

        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "NLI classifier requires transformers and torch or an injected callable."
            ) from e
        nli_pipe = pipeline(
            "text-classification",
            model=self.nli_model,
            top_k=None,
            truncation=True,
            batch_size=16,
        )

        def _hf_nli(pairs: list[tuple[str, str]]) -> list[float]:
            # Proper cross-encoder text-pair input, all pairs in one batched call
            outputs = nli_pipe([{"text": p, "text_pair": h} for p, h in pairs])
            scores = []
            for out in outputs:
                items = out if isinstance(out, list) else [out]
                entail = next(
                    (float(i["score"]) for i in items
                     if isinstance(i, dict) and "entail" in str(i.get("label", "")).lower()),
                    None,
                )
                if entail is None:
                    raise ValueError("NLI model returned no entailment label")
                scores.append(entail)
            return scores

        self._nli_classifier = _hf_nli
        return self._nli_classifier

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        context = context or {}
        source = context.get("source")

        # No source: fail closed by default; "skip" mode passes and records it
        if source is None or not str(source).strip():
            if self.on_missing_source == "skip":
                return GuardrailResult(
                    guardrail=self.name,
                    passed=True,
                    score=1.0,
                    metadata={"skipped": "no_source", "missing_source": True},
                )
            return GuardrailResult(
                guardrail=self.name,
                passed=False,
                score=0.0,
                reason="no source provided: groundedness cannot be verified",
                metadata={"missing_source": True},
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

        # Step 2: Claim filter. Only indices/counts go to metadata, never sentence text.
        claims: list[tuple[int, str]] = []
        skipped_indices: list[int] = []
        for idx, sentence in enumerate(sentences):
            claim = _extract_claim(sentence, self.min_words)
            if claim is None:
                skipped_indices.append(idx)
            else:
                claims.append((idx, claim))

        base_meta: dict[str, Any] = {
            "threshold": self.t_ground,
            "t_low": self.t_low,
            "sentences_total": len(sentences),
            "sentences_skipped": len(skipped_indices),
            "skipped_sentence_indices": skipped_indices,
            "claims_evaluated": len(claims),
        }

        if not claims:
            return GuardrailResult(
                guardrail=self.name,
                passed=True,
                score=1.0,
                metadata={**base_meta, "score": 1.0, "ungrounded_count": 0, "ungrounded_indices": []},
            )

        # Step 3: Chunk source; rank across ALL chunks unless an explicit cap is configured
        chunks = _chunk_text(source_str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
        total_chunks = len(chunks)
        truncated = False
        if self.max_source_chunks is not None and total_chunks > self.max_source_chunks:
            chunks = chunks[:self.max_source_chunks]
            truncated = True

        claim_texts = [c for _, c in claims]
        sim_matrix = self._get_embedder()(claim_texts, chunks)
        if len(sim_matrix) != len(claims) or any(len(row) != len(chunks) for row in sim_matrix):
            raise ValueError("embedder returned a similarity matrix of the wrong shape")

        # Step 4: Cosine only rejects. Collect every NLI pair, then score them in ONE batch.
        nli_pairs: list[tuple[str, str]] = []
        pair_owner: list[int] = []
        best_sims: list[float] = []
        for ci, (_, claim) in enumerate(claims):
            sims = [float(x) for x in sim_matrix[ci]]
            ranked = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)
            best_sims.append(sims[ranked[0]] if ranked else 0.0)
            for j in ranked[:self.top_k]:
                if sims[j] >= self.t_low:
                    nli_pairs.append((chunks[j], claim))
                    pair_owner.append(ci)

        nli_scores: list[float] = []
        if nli_pairs:
            nli_scores = [float(x) for x in self._get_nli()(nli_pairs)]
            if len(nli_scores) != len(nli_pairs):
                raise ValueError("NLI classifier returned the wrong number of scores")

        # Step 5: per-claim score = max entailment over its chunks (0.0 if rejected by cosine)
        per_claim: list[list[float]] = [[] for _ in claims]
        for owner, score in zip(pair_owner, nli_scores):
            per_claim[owner].append(score)

        claim_scores: list[float] = []
        details: list[dict[str, Any]] = []
        for ci, (sent_idx, _) in enumerate(claims):
            evaluated = per_claim[ci]
            score = max(evaluated) if evaluated else 0.0
            claim_scores.append(score)
            details.append({
                "sentence_index": sent_idx,
                "best_sim": round(best_sims[ci], 4),
                "action": "evaluated_by_nli" if evaluated else "rejected_by_cosine",
                "score": round(score, 4),
                "chunks_evaluated": len(evaluated),
            })

        # Step 6: score = min(claims)
        min_score = min(claim_scores)
        ungrounded_indices = [
            claims[ci][0] for ci, sc in enumerate(claim_scores) if sc < self.t_ground
        ]
        passed = min_score >= self.t_ground

        reason = (
            None
            if passed
            else f"groundedness score {min_score:.2f} below threshold {self.t_ground:.2f}: "
                 f"{len(ungrounded_indices)} ungrounded claim(s)"
        )

        return GuardrailResult(
            guardrail=self.name,
            passed=passed,
            score=round(min_score, 4),
            reason=reason,
            metadata={
                **base_meta,
                "score": round(min_score, 4),
                "ungrounded_count": len(ungrounded_indices),
                "ungrounded_indices": ungrounded_indices,
                "details": details,
                "source_chunks_total": total_chunks,
                "source_chunks_ranked": len(chunks),
                "source_truncated": truncated,
            },
        )
