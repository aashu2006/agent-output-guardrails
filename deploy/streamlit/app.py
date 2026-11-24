"""Streamlit demo for agent-output-guardrails.

Validate tab: runs one LLM response through the full engine (schema, PII, toxicity,
groundedness) and shows the policy decision.
Re-ask tab: engine.run() with a scripted FakeProvider (no network calls).
"""

# Keep CPU/memory use low on a ~1GB host. Must run before torch/transformers are imported.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch  # noqa: E402

torch.set_num_threads(1)

import html  # noqa: E402

import streamlit as st  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from guardrails import Action, FakeProvider, GuardrailEngine, PolicyConfig  # noqa: E402

REPO_URL = "https://github.com/aashu2006/agent-output-guardrails"


class Invoice(BaseModel):
    customer: str
    amount: int
    paid: bool


# Inputs reused verbatim from examples/demo.py
GOOD = '{"customer": "Acme Corp", "amount": 4999, "paid": false}'
BAD_JSON = '{"customer": "Acme Corp", "amount": 4999, paid: false'
LEAKY = '{"customer": "Reach me at john@gmail.com or +91 9876543210", "amount": 500, "paid": true}'
TOXIC = '{"customer": "You stupid idiot pay me right now!", "amount": 500, "paid": true}'
RAG_SOURCE = (
    "Acme Corp was founded in 2012 by Sarah Jenkins. "
    "The total invoice balance of $4,999 is due by December 31, 2026."
)
GROUNDED = '{"customer": "Acme Corp was founded by Sarah Jenkins.", "amount": 4999, "paid": false}'
HALLUCINATED = '{"customer": "Acme Corp was acquired by Google for 10 Billion dollars", "amount": 4999, "paid": false}'

EXAMPLES = [
    ("Clean invoice", GOOD, ""),
    ("PII leak", LEAKY, ""),
    ("Broken JSON", BAD_JSON, ""),
    ("Toxic text", TOXIC, ""),
    ("Grounded claim + source", GROUNDED, RAG_SOURCE),
    ("Hallucinated claim + source", HALLUCINATED, RAG_SOURCE),
]

ALL_CHECKS = ["schema", "pii", "toxicity", "groundedness"]
MAX_RETRIES = 2

BADGE_COLORS = {
    Action.ALLOW: "#15803d",
    Action.MODIFY: "#b45309",
    Action.REASK: "#1d4ed8",
    Action.BLOCK: "#b91c1c",
}


def _make_config(enabled: list[str]) -> PolicyConfig:
    # Source context is optional in this demo, so groundedness is skipped (not failed)
    # when it is empty. The library default is to fail closed.
    return PolicyConfig(
        enabled=enabled,
        groundedness_on_missing_source="skip",
        max_retries=MAX_RETRIES,
    )


def _warm_toxicity(engine: GuardrailEngine) -> None:
    engine.toxicity_check._get_classifier()
    r = engine.toxicity_check.check("Thanks, have a great day!")
    if r.error:
        raise RuntimeError(r.reason)


def _warm_groundedness(engine: GuardrailEngine) -> None:
    engine.groundedness_check._get_embedder()
    engine.groundedness_check._get_nli()
    r = engine.groundedness_check.check("Acme Corp was founded in 2012.", context={"source": RAG_SOURCE})
    if r.error:
        raise RuntimeError(r.reason)


@st.cache_resource(show_spinner="Loading and warming up models (first visit only) ...")
def load_engine() -> tuple[GuardrailEngine, dict[str, str]]:
    """Build the engine once per process and warm each model.

    A model that fails to load (missing package, MemoryError, download error) is dropped
    from the enabled checks so the rest of the app keeps working.
    """
    engine = GuardrailEngine(Invoice, config=_make_config(ALL_CHECKS))
    disabled: dict[str, str] = {}
    for name, warm in (("toxicity", _warm_toxicity), ("groundedness", _warm_groundedness)):
        try:
            warm(engine)
        except Exception as exc:  # keep serving with this check off
            disabled[name] = type(exc).__name__
            print(f"[startup] {name} disabled: {type(exc).__name__}", flush=True)
    engine.config = _make_config([c for c in ALL_CHECKS if c not in disabled])
    print(f"[startup] ready, enabled checks: {engine.config.enabled}", flush=True)
    return engine, disabled


def badge(action: Action) -> None:
    st.markdown(
        f'<div style="background:{BADGE_COLORS[action]};color:#fff;font-size:2.2rem;'
        f'font-weight:700;text-align:center;padding:14px;border-radius:10px;letter-spacing:2px">'
        f"{html.escape(action.value.upper())}</div>",
        unsafe_allow_html=True,
    )


def check_notes(name: str, counts: dict) -> str:
    """Human summary from whitelisted counts only (Decision.metadata never holds raw values)."""
    if name == "pii":
        found = counts.get("found", {})
        return ", ".join(f"{k}={v}" for k, v in found.items()) or "nothing found"
    if name == "groundedness":
        if counts.get("missing_source"):
            return "skipped (no source given)"
        if "claims_evaluated" in counts:
            return (f"{counts['claims_evaluated']} claim(s) checked, "
                    f"{counts.get('ungrounded_count', 0)} ungrounded")
    if name == "toxicity" and "threshold" in counts:
        return f"threshold {counts['threshold']}"
    if name == "schema" and counts.get("fence_stripped"):
        return "code fence stripped"
    return ""


def show_output(decision) -> None:
    st.markdown("**Final output**")
    if decision.output is None:
        st.code("(suppressed: unsafe to ship)", language=None)
    else:
        st.code(decision.output, language="json")


def show_validation(decision) -> None:
    badge(decision.action)
    st.markdown(f"**Reason:** {decision.reason or '-'}")
    show_output(decision)

    checks = []
    for name, c in decision.metadata.get("checks", {}).items():
        status = "ERROR" if c["error"] else ("passed" if c["passed"] else "FAILED")
        score = "" if c.get("score") is None else f"{c['score']:.4f}"
        checks.append({"check": name, "status": status, "score": score,
                       "notes": check_notes(name, c.get("counts", {}))})
    st.markdown("**Checks**")
    st.dataframe(checks, hide_index=True, width="stretch")

    order = ["schema", "parallel", "policy", "total"]
    timings = [{"stage": s, "ms": round(decision.timings[s] * 1000, 2)}
               for s in order if s in decision.timings]
    st.markdown("**Timings**")
    st.dataframe(timings, hide_index=True, width="stretch")


class TracingEngine(GuardrailEngine):
    """Records every attempt's input and decision inside engine.run()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[tuple[str, object]] = []

    def validate(self, candidate, *args, **kwargs):
        decision = super().validate(candidate, *args, **kwargs)
        self.trace.append((candidate, decision))
        return decision


def run_reask(engine: GuardrailEngine, responses: list[str]) -> None:
    provider = FakeProvider(list(responses))
    tracer = TracingEngine(
        Invoice,
        config=engine.config,
        provider=provider,
        toxicity_scanner=engine.toxicity_check,  # reuse the loaded models
        groundedness_checker=engine.groundedness_check,
    )
    final = tracer.run("Return the Acme Corp invoice as JSON.")

    badge(final.action)
    st.markdown(
        f"**Final: {final.action.value.upper()}** after {len(tracer.trace)} attempt(s). "
        f"Provider called {provider.call_count}x, re-asks used "
        f"{final.metadata.get('attempts', 0)}/{engine.config.max_retries}."
    )
    st.markdown("**Attempts**")
    st.dataframe(
        [{"attempt": i, "LLM response": resp, "decision": d.action.value.upper(), "reason": d.reason}
         for i, (resp, d) in enumerate(tracer.trace, start=1)],
        hide_index=True,
        width="stretch",
    )
    show_output(final)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Agent Output Guardrails", layout="wide")
st.markdown(
    f"**Agent Output Guardrails**: checks every LLM response (schema, PII, toxicity, groundedness) "
    f"before it reaches the user, then allows, redacts, re-asks or blocks it. [Source]({REPO_URL})"
)

ENGINE, DISABLED = load_engine()
if DISABLED:
    st.caption(
        "Note: " + ", ".join(f"{n} check is off (model failed to load: {e})" for n, e in DISABLED.items())
    )

st.session_state.setdefault("response", GOOD)
st.session_state.setdefault("source", "")


def _load_example(resp: str, src: str) -> None:
    st.session_state["response"] = resp
    st.session_state["source"] = src
    st.session_state["run_validate"] = True


validate_tab, reask_tab = st.tabs(["Validate", "Re-ask loop"])

with validate_tab:
    left, right = st.columns(2)
    with left:
        st.text_area("LLM response (JSON for the Invoice schema)", key="response", height=130)
        st.text_area("Source context (optional, enables groundedness)", key="source", height=90)
        if st.button("Validate", type="primary"):
            st.session_state["run_validate"] = True
        st.markdown("**Examples** (fill the inputs and run)")
        cols = st.columns(3)
        for i, (label, resp, src) in enumerate(EXAMPLES):
            cols[i % 3].button(label, key=f"ex_{i}", on_click=_load_example, args=(resp, src),
                               width="stretch")
        st.caption(
            "Schema: `{customer: str, amount: int, paid: bool}`. "
            "Without a source, groundedness is skipped in this demo."
        )
    with right:
        if st.session_state.pop("run_validate", False):
            decision = ENGINE.validate(
                st.session_state["response"] or "",
                source=(st.session_state["source"] or "").strip() or None,
            )
            show_validation(decision)

with reask_tab:
    st.markdown(
        "`engine.run()` with a scripted `FakeProvider` (no network). Every re-asked response goes "
        f"through the full pipeline again; budget = {MAX_RETRIES} re-asks, then BLOCK."
    )
    c1, c2, _ = st.columns([1, 1, 2])
    fixed = c1.button("Broken then fixed", width="stretch")
    broken = c2.button("Always broken", width="stretch")
    if fixed:
        run_reask(ENGINE, [BAD_JSON, GOOD])
    elif broken:
        run_reask(ENGINE, [BAD_JSON, BAD_JSON, BAD_JSON])
