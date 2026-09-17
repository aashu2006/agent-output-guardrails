from pydantic import BaseModel
from guardrails.checks.schema import SchemaValidator


class Invoice(BaseModel):
    customer: str
    amount: int
    paid: bool


validator = SchemaValidator(Invoice)

good = '{"customer": "Acme Corp", "amount": 4999, "paid": false}'
bad_json = '{"customer": "Acme Corp", "amount": 4999, paid: false'
wrong_type = '{"customer": "Acme Corp", "amount": "a lot", "paid": false}'

for label, response in [("valid", good), ("broken JSON", bad_json), ("wrong type", wrong_type)]:
    r = validator.check(response)
    print(f"\n--- {label} ---")
    print("passed:", r.passed)
    print("reason:", r.reason)

# --- PII demo ---
from guardrails.checks.pii import PIIScanner

pii = PIIScanner()

leaky = '{"customer": "Reach me at john@gmail.com or +91 9876543210", "amount": 500, "paid": true}'

r_schema = validator.check(leaky)
r_pii = pii.check(leaky, context={"parsed": r_schema.metadata.get("parsed")})

print("\n--- PII in a valid response ---")
print("schema passed:", r_schema.passed)
print("pii passed:", r_pii.passed)
print("found:", r_pii.metadata["found"])
print("transformed:", r_pii.transformed_output)

clean = '{"customer": "Acme Corp", "amount": 500, "paid": true}'
r_clean = pii.check(clean, context={"parsed": validator.check(clean).metadata.get("parsed")})

print("\n--- clean response, no PII ---")
print("found:", r_clean.metadata["found"])
print("transformed:", r_clean.transformed_output)