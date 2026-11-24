from typing import Literal

from pydantic import BaseModel


class PolicyConfig(BaseModel):
    enabled: list[str] = ["schema", "pii"]
    pii_categories: list[str] = ["email", "phone", "aadhaar", "pan", "card", "ip"]
    groundedness_threshold: float = 0.7
    groundedness_t_low: float = 0.35
    # groundedness enabled but no source: "block" fails closed (default), "skip" passes
    groundedness_on_missing_source: Literal["block", "skip"] = "block"
    toxicity_threshold: float = 0.8
    max_retries: int = 2
