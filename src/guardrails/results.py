from pydantic import BaseModel, Field

class GuardrailResult(BaseModel):
    guardrail: str
    passed: bool
    error: bool = False
    score: float | None = None
    reason: str | None = None
    transformed_output: str | None = None
    metadata: dict = Field(default_factory=dict)