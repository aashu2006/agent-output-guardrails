import re
import json
from guardrails.base import Guardrail
from guardrails.results import GuardrailResult

# Zero-width / invisible chars. Matching runs on a copy with these removed; redaction is
# applied to the ORIGINAL text, so ZWJ emoji outside a PII span are left intact.
INVISIBLE_CHARS = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")

# order = priority: overlap hone pe upar wala jeet ta hai
PATTERNS = {
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "aadhaar": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "phone": re.compile(
        r"(?<!\d)(?<!\d[ .-])(?:\+91[ .-]?|0)?(?:"
        r"\(\d{3}\)[ .-]?\d{3}[ .-]?\d{4}|"        # (987) 654-3210
        r"\d{10}|"                                  # 9876543210 (contiguous, as before)
        # separated forms must look like a mobile number (starts 6-9) in a phone grouping,
        # so IDs / decimals / dates / versions ("12345 67890", "3.141592653") are not phones
        r"[6-9](?:[ .-]\d){9}|"                     # 9 8 7 6 5 4 3 2 1 0 / 9.8.7...
        r"[6-9]\d{4}[ .-]\d{5}|"                    # 98765 43210
        r"[6-9]\d{2}[ .-]\d{3}[ .-]\d{4}"           # 987-654-3210 / 987.654.3210
        r")(?![ .-]?\d)"
    ),
    "email": re.compile(
        r"(?:"
        r"[\w.+-]+(?:@|\s*(?:\[at\]|\(at\))\s*)[\w-]+(?:(?:\.|\s*(?:\[dot\]|\(dot\))\s*)[\w-]+)+|"
        # worded "john at gmail dot com": local part needs a letter, domain labels start with a
        # letter and the TLD is alphabetic -> "gate 4 at 10 dot 30" is not an email
        r"[\w.+-]*[a-z][\w.+-]*\s+at\s+[a-z][\w-]*(?:\s+dot\s+[a-z][\w-]*)*\s+dot\s+[a-z]{2,}\b"
        r")",
        re.IGNORECASE,
    ),
    "pan": re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b"),
    # exactly 4 octets, not part of a longer dotted run (so "1.2.3.4.5.6.7.8.9.0" is not an IP)
    "ip": re.compile(r"(?<!\d)(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)"),
}

# PAN 4th char = holder type. Uppercase PAN shape is always flagged; a lower/mixed-case
# match needs a valid holder type or a nearby keyword (so "sales2024q" is not a PAN).
PAN_HOLDER_TYPES = frozenset("PCHFATBLJG")
PAN_CONTEXT = re.compile(r"\b(?:pan|tax|taxpayer|identifier|id|permanent\s+account)\b", re.I)


def _pan_plausible(text: str, m: re.Match) -> bool:
    value = m.group()
    if value.isupper():
        return True
    if value[3].upper() in PAN_HOLDER_TYPES:
        return True
    return bool(PAN_CONTEXT.search(text[max(0, m.start() - 40):m.start()]))


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

def _normalize(text: str) -> tuple[str, list[int]]:
    """Copy of text without invisible chars + map from normalized index -> original index."""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in INVISIBLE_CHARS:
            continue
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def _redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Returns (sanitized_text, counts). Match on normalized copy -> overlap drop -> redact original."""
    norm, index_map = _normalize(text)
    spans = []  # (start, end, category) in normalized coordinates
    for category, pattern in PATTERNS.items():
        for m in pattern.finditer(norm):
            if category == "card":
                digits = re.sub(r"[ -]", "", m.group())
                if not _luhn_valid(digits):
                    continue  # Luhn fail = card nahi hai
            if category == "pan" and not _pan_plausible(norm, m):
                continue
            spans.append((m.start(), m.end(), category))

    # overlap drop: pehle aaye (higher priority) jeette hain
    kept = []
    for start, end, cat in spans:
        overlaps = any(s < end and start < e for s, e, _ in kept)
        if not overlaps:
            kept.append((start, end, cat))

    # right-to-left replace on the ORIGINAL text taaki positions shift na hon
    counts: dict[str, int] = {}
    for start, end, cat in sorted(kept, reverse=True):
        o_start, o_end = index_map[start], index_map[end - 1] + 1
        text = text[:o_start] + REPLACEMENTS[cat] + text[o_end:]
        counts[cat] = counts.get(cat, 0) + 1
    return text, counts

def _walk_and_redact(obj, counts):
    """Parsed dict/list mein recursively ghus ke har string field (keys + values) saaf karo."""
    if isinstance(obj, str):
        clean, c = _redact_text(obj)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        return clean
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            clean_k = k
            if isinstance(k, str):
                clean_k, c_k = _redact_text(k)
                for cat, count in c_k.items():
                    counts[cat] = counts.get(cat, 0) + count
            cleaned[clean_k] = _walk_and_redact(v, counts)
        return cleaned
    if isinstance(obj, list):
        return [_walk_and_redact(item, counts) for item in obj]
    return obj  # numbers, bools, None as-is


class PIIScanner(Guardrail):
    name = "pii"

    def _check(self, candidate: str, context: dict | None = None) -> GuardrailResult:
        ctx = context or {}
        parsed = ctx.get("parsed")
        counts: dict[str, int] = {}

        if parsed is not None:
            cleaned = _walk_and_redact(parsed, counts)
            # prose outside a stripped code fence is never shipped, but it is still scanned
            outside = ctx.get("outside_text")
            if outside:
                _, outside_counts = _redact_text(outside)
                for k, v in outside_counts.items():
                    counts[k] = counts.get(k, 0) + v
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