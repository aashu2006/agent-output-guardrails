"""Tests for latency benchmarking harness."""

from pydantic import BaseModel
from benchmarks.latency_bench import (
    BenchmarkSchema,
    run_latency_benchmark,
    SYNTHETIC_PAYLOADS,
)
from guardrails.config import PolicyConfig
from guardrails.engine import GuardrailEngine


def test_latency_benchmark_100_runs():
    config = PolicyConfig(enabled=["pii"])
    engine = GuardrailEngine(schema=BenchmarkSchema, config=config)

    stats = run_latency_benchmark(engine, num_requests=100, p95_threshold_ms=200.0)

    assert stats["num_requests"] == 100
    assert stats["sla_passed"] is True
    assert stats["p95_ms"] < 200.0  # Must comfortably satisfy <200ms SLA
    assert stats["p50_ms"] > 0.0
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]
