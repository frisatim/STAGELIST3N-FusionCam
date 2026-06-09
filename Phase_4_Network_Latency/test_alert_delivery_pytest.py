import pytest

from Phase_4_Network_Latency.alert_delivery_benchmark import (
    AlertLatency,
    run_http_post_benchmark,
    run_queue_benchmark,
    summarize_latencies,
)


def test_queue_benchmark_delivers_all_events():
    rows = run_queue_benchmark(events=5, rate_hz=0)

    assert len(rows) == 5
    assert {row.transport for row in rows} == {"queue"}
    assert all(row.latency_ms >= 0 for row in rows)


def test_http_post_benchmark_delivers_all_events():
    try:
        rows = run_http_post_benchmark(events=3, rate_hz=0)
    except PermissionError:
        pytest.skip("Local sockets are blocked in this sandbox")

    assert len(rows) == 3
    assert {row.transport for row in rows} == {"http_post"}


def test_summarize_latencies_reports_p95():
    summary = summarize_latencies(
        [
            AlertLatency(1, "queue", 1.0),
            AlertLatency(2, "queue", 3.0),
            AlertLatency(3, "queue", 5.0),
        ]
    )

    assert summary["events"] == 3.0
    assert summary["median_ms"] == 3.0
    assert summary["p95_ms"] == 5.0
