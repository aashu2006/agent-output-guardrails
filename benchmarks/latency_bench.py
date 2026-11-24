"""Latency benchmarking harness for GuardrailEngine and individual check components.

Measures end-to-end p50/p90/p95/p99 plus p50 and p95 for every pipeline stage
(schema, parallel checks, policy, total) across synthetic LLM workloads.
Runs PII, toxicity and groundedness; ML stages whose libraries or models are not
installed locally are skipped with a clear message (nothing is downloaded).
Verifies against latency budgets (target p95 < 200ms).
"""

import argparse
import importlib.util
import statistics
import time
from typing import Any

from pydantic import BaseModel

from guardrails.checks.groundedness import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NLI_MODEL,
    GroundednessChecker,
)
from guardrails.checks.toxicity import ToxicityScanner
from guardrails.config import PolicyConfig
from guardrails.engine import GuardrailEngine


class BenchmarkSchema(BaseModel):
    user_id: str
    message: str
    score: float


SYNTHETIC_PAYLOADS = [
    # 1. Clean valid JSON
    '{"user_id": "usr_1001", "message": "Everything is running smoothly.", "score": 0.95}',
    # 2. PII spaced digits (adversarial red-team case)
    '{"user_id": "usr_1002", "message": "Contact support at 9 8 7 6 5 4 3 2 1 0 today.", "score": 0.88}',
    # 3. PII obfuscated email (adversarial red-team case)
    '{"user_id": "usr_1003", "message": "Send feedback to ops [at] acme [dot] com promptly.", "score": 0.91}',
    # 4. Markdown-fenced valid JSON
    '```json\n{"user_id": "usr_1004", "message": "Fenced payload processing test.", "score": 0.85}\n```',
    # 5. Regular email + phone
    '{"user_id": "usr_1005", "message": "Call +91 9876543210 or email test@example.com", "score": 0.99}',
    # 6. Moderate length text
    '{"user_id": "usr_1006", "message": "Agent output guardrails provide deterministic security controls for high-throughput enterprise pipelines.", "score": 0.92}',
]

# RAG source used when the groundedness stage is enabled
BENCH_SOURCE = (
    "All systems are running smoothly after the latest deployment. "
    "Customers can contact support by phone or email for help with their accounts. "
    "Feedback should be sent to the operations team promptly. "
    "Agent output guardrails provide deterministic security controls for high-throughput enterprise pipelines."
)

STAGES = ("schema", "parallel", "policy", "total")

ML_STAGE_REQUIREMENTS = {
    "toxicity": (["transformers", "torch"], ["unitary/toxic-bert"]),
    "groundedness": (
        ["sentence_transformers", "transformers", "torch"],
        [DEFAULT_EMBEDDING_MODEL, DEFAULT_NLI_MODEL],
    ),
}


def stage_available(stage: str) -> tuple[bool, str]:
    """(available, reason). Checks libraries are importable and models are in the local HF cache."""
    libs, models = ML_STAGE_REQUIREMENTS[stage]
    for lib in libs:
        if importlib.util.find_spec(lib) is None:
            return False, f"python package '{lib}' not installed (pip install -e '.[ml]')"
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False, "huggingface_hub not installed"
    for model in models:
        if not isinstance(try_to_load_from_cache(model, "config.json"), str):
            return False, f"model '{model}' not in local HuggingFace cache"
    return True, "ok"


def build_engine(
    stages: tuple[str, ...] = ("pii", "toxicity", "groundedness"),
) -> tuple[GuardrailEngine, list[str], dict[str, str]]:
    """Returns (engine, enabled_stages, skipped_stages{stage: reason})."""
    enabled: list[str] = []
    skipped: dict[str, str] = {}
    for stage in stages:
        if stage in ML_STAGE_REQUIREMENTS:
            ok, why = stage_available(stage)
            if not ok:
                skipped[stage] = why
                continue
        enabled.append(stage)

    config = PolicyConfig(enabled=["schema", *enabled])
    engine = GuardrailEngine(
        schema=BenchmarkSchema,
        config=config,
        toxicity_scanner=ToxicityScanner(threshold=config.toxicity_threshold),
        groundedness_checker=GroundednessChecker(
            t_low=config.groundedness_t_low, t_ground=config.groundedness_threshold
        ),
    )
    return engine, enabled, skipped


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    return data[f] * (c - k) + data[c] * (k - f) if c != f else data[f]


def run_latency_benchmark(
    engine: GuardrailEngine,
    payloads: list[str] | None = None,
    num_requests: int = 100,
    p95_threshold_ms: float = 200.0,
    source: str | None = None,
) -> dict[str, Any]:
    """Runs synthetic requests through the engine and calculates latency statistics."""
    inputs = payloads or SYNTHETIC_PAYLOADS
    if source is None and "groundedness" in engine.config.enabled:
        source = BENCH_SOURCE

    latencies_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {stage: [] for stage in STAGES}

    # Warmup: every payload once (loads lazy ML models outside the measured loop)
    for payload in inputs:
        engine.validate(payload, source=source)

    for i in range(num_requests):
        payload = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        decision = engine.validate(payload, source=source)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        for stage in STAGES:
            if stage in decision.timings:
                stage_ms[stage].append(decision.timings[stage] * 1000.0)

    latencies_sorted = sorted(latencies_ms)
    p95 = percentile(latencies_sorted, 0.95)

    stages: dict[str, dict[str, float]] = {}
    for stage, values in stage_ms.items():
        values.sort()
        stages[stage] = {
            "n": len(values),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
        }

    return {
        "num_requests": num_requests,
        "enabled_checks": [c for c in engine.config.enabled if c != "schema"],
        "mean_ms": statistics.mean(latencies_sorted),
        "p50_ms": percentile(latencies_sorted, 0.50),
        "p90_ms": percentile(latencies_sorted, 0.90),
        "p95_ms": p95,
        "p99_ms": percentile(latencies_sorted, 0.99),
        "min_ms": latencies_sorted[0],
        "max_ms": latencies_sorted[-1],
        "stages": stages,
        "target_p95_ms": p95_threshold_ms,
        "sla_passed": p95 <= p95_threshold_ms,
    }


def print_benchmark_report(stats: dict[str, Any], skipped: dict[str, str] | None = None) -> None:
    """Prints a structured ASCII report of benchmark latency results."""
    status = "PASSED" if stats["sla_passed"] else "FAILED"
    print("\n" + "=" * 65)
    print("      AGENT OUTPUT GUARDRAILS - LATENCY BENCHMARK REPORT      ")
    print("=" * 65)
    print(f"Requests Evaluated: {stats['num_requests']}")
    print(f"Checks Enabled    : schema, {', '.join(stats['enabled_checks']) or '(none)'}")
    for stage, why in (skipped or {}).items():
        print(f"Skipped Stage     : {stage} -> {why}")
    print(f"p95 Latency Target: {stats['target_p95_ms']:.2f} ms")
    print(f"SLA Verification  : [{status}]")
    print("-" * 65)
    print(f"{'Metric':<25} {'Value (ms)':<15} {'Target (ms)':<15}")
    print("-" * 65)
    print(f"{'p50 (Median)':<25} {stats['p50_ms']:<15.3f} {'--':<15}")
    print(f"{'p90':<25} {stats['p90_ms']:<15.3f} {'--':<15}")
    print(f"{'p95':<25} {stats['p95_ms']:<15.3f} {stats['target_p95_ms']:<15.2f}")
    print(f"{'p99':<25} {stats['p99_ms']:<15.3f} {'--':<15}")
    print(f"{'Mean':<25} {stats['mean_ms']:<15.3f} {'--':<15}")
    print(f"{'Min':<25} {stats['min_ms']:<15.3f} {'--':<15}")
    print(f"{'Max':<25} {stats['max_ms']:<15.3f} {'--':<15}")
    print("-" * 65)
    print(f"{'Stage':<25} {'p50 (ms)':<15} {'p95 (ms)':<15}")
    print("-" * 65)
    for stage in STAGES:
        s = stats["stages"][stage]
        if s["n"] == 0:
            print(f"{stage:<25} {'(not run)':<15} {'':<15}")
        else:
            print(f"{stage:<25} {s['p50_ms']:<15.3f} {s['p95_ms']:<15.3f}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GuardrailEngine Latency Benchmark Harness")
    parser.add_argument("--requests", type=int, default=100, help="Number of synthetic requests (default: 100)")
    parser.add_argument("--target-p95", type=float, default=200.0, help="Target p95 latency threshold in ms (default: 200.0)")
    args = parser.parse_args()

    engine, _, skipped = build_engine()
    for stage, why in skipped.items():
        print(f"[benchmark] skipping {stage} stage: {why}")

    stats = run_latency_benchmark(
        engine=engine,
        num_requests=args.requests,
        p95_threshold_ms=args.target_p95,
    )
    print_benchmark_report(stats, skipped)


if __name__ == "__main__":
    main()
