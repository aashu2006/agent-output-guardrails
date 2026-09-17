import re
import json
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

# order = priority: overlap hone pe upar wala jeet ta hai
PATTERNS = {
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "aadhaar": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+91[ -]?)?(?:0)?\d{10}\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

REPLACEMENTS = {
    "card": "[REDACTED_CARD]",
    "aadhaar": "[REDACTED_ID]",
    "phone": "[REDACTED_PHONE]",
    "email": "[REDACTED_EMAIL]",
    "pan": "[REDACTED_ID]",
    "ip": "[REDACTED_IP]",
}

def _luhn_valid(digits: str) -> bool:
    """Card checksum. Right se har doosra digit double, >9 toh -9, sum %10 == 0."""
    nums = [int(d) for d in digits]
    nums.reverse()
    total = 0
    for i, d in enumerate(nums):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

def _redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Returns (sanitized_text, counts). Span collect -> overlap drop by priority -> right-to-left replace."""
    spans = []  # (start, end, category)
    for category, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            if category == "card":
                digits = re.sub(r"[ -]", "", m.group())
                if not _luhn_valid(digits):
                    continue  # Luhn fail = card nahi hai
            spans.append((m.start(), m.end(), category))

    # overlap drop: pehle aaye (higher priority) jeette hain
    kept = []
    for start, end, cat in spans:
        overlaps = any(s < end and start < e for s, e, _ in kept)
        if not overlaps:
            kept.append((start, end, cat))

    # right-to-left replace taaki positions shift na hon
    counts: dict[str, int] = {}
    for start, end, cat in sorted(kept, reverse=True):
        text = text[:start] + REPLACEMENTS[cat] + text[end:]
        counts[cat] = counts.get(cat, 0) + 1
    return text, counts

def _walk_and_redact(obj, counts):
    """Parsed dict/list mein recursively ghus ke har string field saaf karo."""
    if isinstance(obj, str):
        clean, c = _redact_text(obj)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        return clean
    if isinstance(obj, dict):
        return {k: _walk_and_redact(v, counts) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_redact(item, counts) for item in obj]
    return obj  # numbers, bools, None as-is


class PIIScanner(Guardrail):
    name = "pii"

    def check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        parsed = (context or {}).get("parsed")
        counts: dict[str, int] = {}

        if parsed is not None:
            cleaned = _walk_and_redact(parsed, counts)
            transformed = json.dumps(cleaned) if counts else None
        else:
            # fallback: raw text pe (schema ke bina standalone use)
            cleaned_text, counts = _redact_text(candidate)
            transformed = cleaned_text if counts else None

        return GuardrailResult(
            guardrail=self.name,
            passed=True,  # PII milna failure nahi, modify case hai
            transformed_output=transformed,
            metadata={"found": counts},  # sirf counts, values kabhi nahi
        )