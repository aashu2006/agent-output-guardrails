"""Tests for latency benchmarking harness."""

import pytest

from benchmarks import latency_bench
from benchmarks.latency_bench import (
    STAGES,
    BenchmarkSchema,
    build_engine,
    run_latency_benchmark,
)
from guardrails.config import PolicyConfig
from guardrails.engine import GuardrailEngine


def _pii_engine():
    return GuardrailEngine(schema=BenchmarkSchema, config=PolicyConfig(enabled=["pii"]))


def test_latency_benchmark_report_shape():
    """Default suite: structure only, no wall-clock assertions."""
    stats = run_latency_benchmark(_pii_engine(), num_requests=20)

    assert stats["num_requests"] == 20
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    assert set(stats["stages"]) == set(STAGES)
    for stage in STAGES:
        s = stats["stages"][stage]
        assert set(s) == {"n", "p50_ms", "p95_ms"}
        assert s["n"] == 20  # every stage, including policy, is really timed
        assert 0.0 < s["p50_ms"] <= s["p95_ms"]
    assert isinstance(stats["sla_passed"], bool)


def test_build_engine_skips_unavailable_ml_stages(monkeypatch):
    monkeypatch.setattr(latency_bench, "stage_available", lambda stage: (False, "not installed"))
    engine, enabled, skipped = build_engine()

    assert enabled == ["pii"]
    assert skipped == {"toxicity": "not installed", "groundedness": "not installed"}
    assert engine.config.enabled == ["schema", "pii"]


def test_build_engine_enables_all_stages_when_available(monkeypatch):
    monkeypatch.setattr(latency_bench, "stage_available", lambda stage: (True, "ok"))
    engine, enabled, skipped = build_engine()

    assert enabled == ["pii", "toxicity", "groundedness"]
    assert skipped == {}


@pytest.mark.slow
def test_latency_benchmark_p95_under_200ms():
    stats = run_latency_benchmark(_pii_engine(), num_requests=100, p95_threshold_ms=200.0)

    assert stats["sla_passed"] is True
    assert stats["p95_ms"] < 200.0  # Must comfortably satisfy <200ms SLA
