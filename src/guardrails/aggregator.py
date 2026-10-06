from pydantic import BaseModel

from guardrails.results import GuardrailResult

class AggregatedResults(BaseModel):
    results: list[GuardrailResult]

    def get(self,name: str) -> GuardrailResult | None:
        for r in self.results:
            if r.guardrail == name:
                return r
        return None

    @property
    def transformed_output(self) -> str | None:
        for r in self.results:
            if r.transformed_output is not None:
                return r.transformed_output
        return None
        
            