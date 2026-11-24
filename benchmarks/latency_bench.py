"""Latency benchmarking harness for GuardrailEngine and individual check components.

Measures p50, p90, p95, and p99 latency distributions across synthetic LLM workloads.
Verifies against latency budgets (target p95 < 200ms).
"""

import argparse
import statistics
import time
from typing import Any

from pydantic import BaseModel

from guardrails.checks.groundedness import GroundednessChecker
from guardrails.checks.pii import PIIScanner
from guardrails.checks.schema import SchemaValidator
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


def run_latency_benchmark(
    engine: GuardrailEngine,
    payloads: list[str] | None = None,
    num_requests: int = 100,
    p95_threshold_ms: float = 200.0,
) -> dict[str, Any]:
    """Runs synthetic requests through the engine and calculates latency statistics."""
    inputs = payloads or SYNTHETIC_PAYLOADS
    latencies_ms: list[float] = []
    schema_ms: list[float] = []
    parallel_ms: list[float] = []
    policy_ms: list[float] = []

    # Warmup run (5 requests)
    for i in range(min(5, len(inputs))):
        engine.validate(inputs[i % len(inputs)])

    # Benchmark loop
    for i in range(num_requests):
        payload = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        decision = engine.validate(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        timings = getattr(decision, "timings", {}) or decision.metadata.get("timings", {})
        if "schema" in timings:
            schema_ms.append(timings["schema"] * 1000.0)
        if "parallel" in timings:
            parallel_ms.append(timings["parallel"] * 1000.0)
        elif "parallel_stage" in timings:
            parallel_ms.append(timings["parallel_stage"] * 1000.0)
        if "policy" in timings:
            policy_ms.append(timings["policy"] * 1000.0)

    latencies_sorted = sorted(latencies_ms)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = min(f + 1, len(data) - 1)
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return d0 + d1

    p50 = percentile(latencies_sorted, 0.50)
    p90 = percentile(latencies_sorted, 0.90)
    p95 = percentile(latencies_sorted, 0.95)
    p99 = percentile(latencies_sorted, 0.99)
    mean_val = statistics.mean(latencies_sorted)

    return {
        "num_requests": num_requests,
        "mean_ms": mean_val,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": latencies_sorted[0],
        "max_ms": latencies_sorted[-1],
        "schema_p50_ms": percentile(sorted(schema_ms), 0.50) if schema_ms else 0.0,
        "parallel_p50_ms": percentile(sorted(parallel_ms), 0.50) if parallel_ms else 0.0,
        "policy_p50_ms": percentile(sorted(policy_ms), 0.50) if policy_ms else 0.0,
        "target_p95_ms": p95_threshold_ms,
        "sla_passed": p95 <= p95_threshold_ms,
    }


def print_benchmark_report(stats: dict[str, Any]) -> None:
    """Prints a structured ASCII report of benchmark latency results."""
    status = "PASSED" if stats["sla_passed"] else "FAILED"
    print("\n" + "=" * 65)
    print("      AGENT OUTPUT GUARDRAILS - LATENCY BENCHMARK REPORT      ")
    print("=" * 65)
    print(f"Requests Evaluated: {stats['num_requests']}")
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
    print("Stage Breakdown (p50):")
    print(f"  • Schema Gate   : {stats['schema_p50_ms']:.3f} ms")
    print(f"  • Parallel Stage: {stats['parallel_p50_ms']:.3f} ms")
    print(f"  • Policy Engine : {stats['policy_p50_ms']:.3f} ms")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GuardrailEngine Latency Benchmark Harness")
    parser.add_argument("--requests", type=int, default=100, help="Number of synthetic requests (default: 100)")
    parser.add_argument("--target-p95", type=float, default=200.0, help="Target p95 latency threshold in ms (default: 200.0)")
    args = parser.parse_args()

    # Fast deterministic config with mock toxicity/groundedness for pure pipeline overhead measurement
    config = PolicyConfig(enabled=["pii"])
    engine = GuardrailEngine(
        schema=BenchmarkSchema,
        config=config,
    )

    stats = run_latency_benchmark(
        engine=engine,
        num_requests=args.requests,
        p95_threshold_ms=args.target_p95,
    )
    print_benchmark_report(stats)


if __name__ == "__main__":
    main()
