from pydantic import BaseModel
from guardrails.checks.schema import SchemaValidator
from guardrails.checks.pii import PIIScanner
from guardrails.engine import GuardrailEngine
from guardrails.config import PolicyConfig
from guardrails.policy import Action
from guardrails.providers import FakeProvider

# ---------------------------------------------------------------------------
# Beautiful Terminal UI Styling & Helper Functions
# ---------------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_section_header(title: str, subtitle: str):
    width = 62
    print(f"\n{CYAN}┌{'─' * width}┐")
    print(f"│ {title:<{width - 1}}│")
    print(f"│ {subtitle:<{width - 1}}│")
    print(f"└{'─' * width}┘{RESET}\n")


def print_card(case_num: str, title: str, input_str: str, d):
    disp_input = input_str[:76] + "..." if len(input_str) > 79 else input_str

    if d.action == Action.MODIFY:
        disp_action = f"{YELLOW}✎ MODIFY (Redacted){RESET}"
    elif d.action == Action.ALLOW:
        disp_action = f"{GREEN}✔ ALLOW{RESET}"
    elif d.action == Action.BLOCK:
        disp_action = f"{RED}✖ BLOCK{RESET}"
    else:
        disp_action = str(d.action.value.upper())

    if d.output is None:
        disp_output = f"{RED}(Suppressed / None - Unsafe to ship){RESET}"
    else:
        out_text = d.output[:76] + "..." if len(d.output) > 79 else d.output
        disp_output = f"{GREEN}{out_text}{RESET}"

    t = d.timings
    schema_ms = t.get("schema", 0.0) * 1000
    total_ms = t.get("total", 0.0) * 1000
    if "parallel" in t:
        parallel_ms = t.get("parallel", 0.0) * 1000
        disp_latency = (
            f"schema: {schema_ms:.1f}ms | parallel: {parallel_ms:.1f}ms | "
            f"total: {total_ms:.1f}ms (Total: {total_ms:.2f}ms)"
        )
    else:
        disp_latency = f"schema: {schema_ms:.1f}ms | total: {total_ms:.1f}ms (Total: {total_ms:.2f}ms)"

    print(f"┌── [{case_num}] {title}")
    print(f"│   Input    : {disp_input}")
    print(f"│   Decision : {disp_action}")
    print(f"│   Reason   : {d.reason}")
    print(f"│   Output   : {disp_output}")
    print(f"│   Latency  : {disp_latency}")
    print(f"└──\n")


class Invoice(BaseModel):
    customer: str
    amount: int
    paid: bool


# ===========================================================================
# 1. INDIVIDUAL GUARDRAILS (Unit Level Checks)
# ===========================================================================
print_section_header(
    "1. INDIVIDUAL GUARDRAILS (Unit Level Checks)",
    "SchemaValidator & PIIScanner in Isolation",
)

validator = SchemaValidator(Invoice)

good = '{"customer": "Acme Corp", "amount": 4999, "paid": false}'
bad_json = '{"customer": "Acme Corp", "amount": 4999, paid: false'
wrong_type = '{"customer": "Acme Corp", "amount": "a lot", "paid": false}'

# [1.A] Valid Schema
r1 = validator.check(good)
print(f"┌── [1.A] VALID SCHEMA PAYLOAD")
print(f"│   Input    : {good}")
print(f"│   Status   : {GREEN}✔ PASSED{RESET}")
print(f"│   Reason   : None (Syntax & Types Valid)")
print(f"└──\n")

# [1.B] Broken JSON Syntax
r2 = validator.check(bad_json)
print(f"┌── [1.B] BROKEN JSON SYNTAX")
print(f"│   Input    : {bad_json}")
print(f"│   Status   : {RED}✖ FAILED{RESET}")
print(f"│   Reason   : {r2.reason}")
print(f"└──\n")

# [1.C] Wrong Data Type
r3 = validator.check(wrong_type)
print(f"┌── [1.C] WRONG DATA TYPE (Type Mismatch)")
print(f"│   Input    : {wrong_type}")
print(f"│   Status   : {RED}✖ FAILED{RESET}")
print(f"│   Reason   : {r3.reason}")
print(f"└──\n")

# --- PII Unit Checks ---
pii = PIIScanner()

leaky = '{"customer": "Reach me at john@gmail.com or +91 9876543210", "amount": 500, "paid": true}'
r_schema = validator.check(leaky)
r_pii = pii.check(leaky, context={"parsed": r_schema.metadata.get("parsed")})

disp_leaky_out = r_pii.transformed_output[:76] + "..." if len(r_pii.transformed_output) > 79 else r_pii.transformed_output
print(f"┌── [1.D] PII LEAK IN VALID RESPONSE")
print(f"│   Input    : {leaky[:76]}...")
print(f"│   Found    : {r_pii.metadata['found']}")
print(f"│   Status   : {YELLOW}✎ REDACTED{RESET}")
print(f"│   Output   : {GREEN}{disp_leaky_out}{RESET}")
print(f"└──\n")

clean = '{"customer": "Acme Corp", "amount": 500, "paid": true}'
r_clean = pii.check(clean, context={"parsed": validator.check(clean).metadata.get("parsed")})
print(f"┌── [1.E] CLEAN RESPONSE (No PII)")
print(f"│   Input    : {clean}")
print(f"│   Found    : {r_clean.metadata['found']} (Clean)")
print(f"│   Status   : {GREEN}✔ UNTOUCHED{RESET}")
print(f"│   Output   : {GREEN}{clean}{RESET}")
print(f"└──\n")


# ===========================================================================
# 2. END-TO-END PIPELINE (GuardrailEngine)
# ===========================================================================
print_section_header(
    "2. END-TO-END PIPELINE (GuardrailEngine)",
    "Schema Gate -> Parallel Scanners -> Policy Decision",
)

engine = GuardrailEngine(Invoice)

# [2.1] Leaky Customer Data
d1 = engine.validate(leaky)
print_card("2.1", "LEAKY CUSTOMER DATA", leaky, d1)

# [2.2] Clean Normal Invoice
d2 = engine.validate(good)
print_card("2.2", "CLEAN NORMAL INVOICE", good, d2)

# [2.3] Corrupted Malformed JSON
d3 = engine.validate(bad_json)
print_card("2.3", "CORRUPTED MALFORMED JSON", bad_json, d3)


# ===========================================================================
# 3. PHASE 2: REAL ML TOXICITY GATE
# ===========================================================================
print_section_header(
    "3. PHASE 2: REAL ML TOXICITY GATE",
    "Live HuggingFace unitary/toxic-bert Inference",
)

toxic_input = '{"customer": "You stupid idiot pay me right now!", "amount": 500, "paid": true}'
polite_input = '{"customer": "Thanks, have a great day!", "amount": 500, "paid": true}'

engine_v2 = GuardrailEngine(
    Invoice,
    config=PolicyConfig(enabled=["schema", "pii", "toxicity"]),
)

try:
    engine_v2.toxicity_check._get_classifier()  # load the real model up front
    model_loaded = True
except ImportError:
    print(f"{YELLOW}toxicity model not installed, run: pip install -e '.[ml]'{RESET}\n")
    model_loaded = False
except Exception as e:
    print(f"{YELLOW}toxicity model failed to load ({type(e).__name__}), skipping section 3{RESET}\n")
    model_loaded = False

if model_loaded:
    d4 = engine_v2.validate(toxic_input)
    print_card("3.1", "ABUSIVE/TOXIC AGENT RESPONSE (expect BLOCK)", toxic_input, d4)

    d5 = engine_v2.validate(polite_input)
    print_card("3.2", "POLITE AGENT RESPONSE (expect ALLOW)", polite_input, d5)

# ===========================================================================
# 4. PHASE 2: GROUNDEDNESS & HALLUCINATION GATE (RAG Verification)
# ===========================================================================
print_section_header(
    "4. PHASE 2: GROUNDEDNESS GATE (RAG Hallucination Detection)",
    "Sentence Splitting -> Cosine Rejection (<T_low) -> NLI Verification",
)

rag_source = (
    "Acme Corp was founded in 2012 by Sarah Jenkins. "
    "The total invoice balance of $4,999 is due by December 31, 2026."
)

grounded_response = '{"customer": "Acme Corp was founded by Sarah Jenkins.", "amount": 4999, "paid": false}'
hallucinated_response = '{"customer": "Acme Corp was acquired by Google for 10 Billion dollars", "amount": 4999, "paid": false}'

engine_v3 = GuardrailEngine(
    Invoice,
    config=PolicyConfig(enabled=["schema", "pii", "groundedness"]),
)

try:
    # load the bi-encoder + NLI models up front (and run one warm-up check),
    # so the timings below don't include model loading
    engine_v3.groundedness_check._get_embedder()
    engine_v3.groundedness_check._get_nli()
    engine_v3.groundedness_check.check(
        "Acme Corp was founded in 2012.", context={"source": rag_source}
    )
    grounding_loaded = True
except ImportError:
    print(f"{YELLOW}groundedness models not installed, run: pip install -e '.[ml]'{RESET}\n")
    grounding_loaded = False
except Exception as e:
    print(f"{YELLOW}groundedness models failed to load ({type(e).__name__}), skipping section 4{RESET}\n")
    grounding_loaded = False

if grounding_loaded:
    d6 = engine_v3.validate(grounded_response, source=rag_source)
    print_card("4.1", "GROUNDED RAG RESPONSE (expect ALLOW)", grounded_response, d6)

    d7 = engine_v3.validate(hallucinated_response, source=rag_source)
    print_card("4.2", "HALLUCINATED RESPONSE (expect BLOCK)", hallucinated_response, d7)

# ===========================================================================
# 5. RE-ASK LOOP (engine.run with FakeProvider, no network)
# ===========================================================================
print_section_header(
    "5. RE-ASK LOOP (engine.run + FakeProvider)",
    "Generate -> Validate -> Re-ask with feedback -> Budget -> Block",
)


class TracingEngine(GuardrailEngine):
    """Demo-only: records the decision of every attempt inside engine.run()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def validate(self, *args, **kwargs):
        decision = super().validate(*args, **kwargs)
        self.trace.append(decision)
        return decision


def print_attempts(engine, provider, final):
    for i, d in enumerate(engine.trace, start=1):
        if d.action == Action.REASK:
            mark = f"{YELLOW}↻ REASK{RESET}"
        elif d.action == Action.BLOCK:
            mark = f"{RED}✖ BLOCK{RESET}"
        else:
            mark = f"{GREEN}✔ {d.action.value.upper()}{RESET}"
        print(f"│   Attempt {i}: {mark}  {d.reason}")
    if final.output is None:
        out = f"{RED}(Suppressed / None - Unsafe to ship){RESET}"
    else:
        out = f"{GREEN}{final.output}{RESET}"
    print(f"│   Final    : {final.action.value.upper()}")
    print(f"│   Output   : {out}")
    print(
        f"│   Attempts : {len(engine.trace)} validated, provider called {provider.call_count}x, "
        f"re-asks used {final.metadata['attempts']}/{engine.config.max_retries}"
    )
    print(f"└──\n")


reask_config = PolicyConfig(enabled=["schema", "pii"], max_retries=2)

# [5.1] broken JSON first, valid JSON after one re-ask
provider_a = FakeProvider([bad_json, good])
engine_a = TracingEngine(Invoice, config=reask_config, provider=provider_a)
final_a = engine_a.run("Return the Acme Corp invoice as JSON.")
print(f"┌── [5.1] BROKEN JSON, THEN FIXED ON RE-ASK (expect ALLOW on attempt 2)")
print_attempts(engine_a, provider_a, final_a)

# [5.2] provider never fixes it: budget runs out -> fail closed
provider_b = FakeProvider([bad_json, bad_json, bad_json])
engine_b = TracingEngine(Invoice, config=reask_config, provider=provider_b)
final_b = engine_b.run("Return the Acme Corp invoice as JSON.")
print(f"┌── [5.2] ALWAYS BROKEN JSON (expect BLOCK after budget runs out)")
print_attempts(engine_b, provider_b, final_b)

print(f"{BOLD}{GREEN}✔ All pipeline checks executed successfully!{RESET}\n")
