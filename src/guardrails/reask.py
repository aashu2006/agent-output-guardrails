import re
from typing import Any
from guardrails.providers import LLMProvider


def _sanitize_untrusted_prompt_text(text: str) -> str:
    """Neutralizes spoofed delimiter headers in untrusted model outputs to prevent prompt injection."""
    sanitized = text.strip()
    return re.sub(r"\[([A-Z_ ]+)\]", r"(\1)", sanitized)


class ReaskEngine:
    """Manages the bounded self-correction loop when a guardrail fails with Action.REASK."""

    def __init__(self, provider: LLMProvider, max_retries: int = 2):
        self.provider = provider
        self.max_retries = max(0, max_retries)
        self.attempts: int = 0

    @property
    def retries_left(self) -> int:
        """Returns the number of retry attempts remaining."""
        return max(0, self.max_retries - self.attempts)

    def build_repair_prompt(
        self,
        original_prompt: str,
        feedback: str,
        source: str | None = None,
        previous_candidate: str | None = None,
    ) -> str:
        """Constructs an unambiguous repair prompt with original context and actionable feedback."""
        safe_feedback = _sanitize_untrusted_prompt_text(feedback)
        parts = [
            original_prompt.strip(),
            "\n[VALIDATION FEEDBACK]",
            f"Your previous response failed validation:\n{safe_feedback}",
        ]
        if previous_candidate and str(previous_candidate).strip():
            safe_candidate = _sanitize_untrusted_prompt_text(str(previous_candidate))
            parts.extend([
                "\n[PREVIOUS RESPONSE]",
                safe_candidate,
            ])
        parts.append(
            "Please return a corrected response strictly adhering to the required schema and constraints."
        )
        if source and str(source).strip():
            safe_source = str(source).strip()
            parts.extend([
                "\n[REFERENCE CONTEXT]",
                safe_source,
                "Ensure your entire response is strictly grounded in this reference context.",
            ])
        return "\n".join(parts)

    def reask(
        self,
        original_prompt: str,
        feedback: str,
        source: str | None = None,
        previous_candidate: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Increments attempt counter, builds repair prompt, and requests a corrected completion."""
        if self.retries_left <= 0:
            raise RuntimeError(
                f"Cannot reask: retry budget exhausted ({self.attempts}/{self.max_retries} attempts made)"
            )
        self.attempts += 1
        repair_prompt = self.build_repair_prompt(
            original_prompt=original_prompt,
            feedback=feedback,
            source=source,
            previous_candidate=previous_candidate,
        )
        return self.provider.complete(repair_prompt, **kwargs)
