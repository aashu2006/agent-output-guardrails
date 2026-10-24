from pydantic import BaseModel


class PolicyConfig(BaseModel):
    enabled: list[str] = ["schema", "pii"]
    pii_categories: list[str] = ["email", "phone", "aadhaar", "pan", "card", "ip"]
    groundedness_threshold: float = 0.7
    toxicity_threshold: float = 0.8
    max_retries: int = 2