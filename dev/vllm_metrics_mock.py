#!/usr/bin/env python3
"""Local vLLM V1 /metrics mock for the bk-lite SLO dashboard.

This worktree does not contain ``dev/mock_metrics.py`` / ``k8s_mock.py`` /
``mock_instances.yaml`` (plugin tests skip those when absent). Local instance
``vllm-8000`` is a Telegraf ``inputs.prometheus`` scrape of :8000 - gauges
already work, so the live-scrape path is the one to fill.

Serves Prometheus text on 0.0.0.0:8000 (no GPU). Telegraf (metric_version=1)
then stores:

    gauge     vllm:num_requests_running          -> vllm:num_requests_running_gauge
    counter   vllm:prompt_tokens_total           -> vllm:prompt_tokens_total_counter
    histogram vllm:time_to_first_token_seconds   -> _count / _sum / _<le>

Histogram bucket ``+Inf`` is **not** stored as ``_+Inf``; the dashboard uses
``_count`` as the +Inf bucket. ``rate()`` / ``histogram_quantile`` need
monotonic growth across scrapes - a static 0 keeps QPS/TPM/P95 at ``--``.

Usage (repo root)::

    python3 dev/vllm_metrics_mock.py
    python3 dev/vllm_metrics_mock.py --dump
    python3 dev/vllm_metrics_mock.py --dump-telegraf --instance-id vllm-8000

No plugin_init. Restart this process if :8000 was already bound. Wait two
Telegraf intervals (default 60s, about 2 min) so ``rate(...[5m])`` is defined.

If ``dev/mock_metrics.py`` writes VictoriaMetrics directly, import
``iter_telegraf_named_series`` from this file and emit it every cycle - do not
copy a one-shot snapshot.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable
from urllib.parse import urlparse

# vLLM V1 histogram bounds from vllm/v1/metrics/loggers.py (main).
TTFT_BUCKETS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
]
ITL_BUCKETS = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
]
REQUEST_LATENCY_BUCKETS = [
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
]

# Eight dashboard histogram families (queue / TTFT / TPOT / token length).
HISTOGRAMS = (
    # (prom_name, buckets, lognormal median, p95)
    ("vllm:time_to_first_token_seconds", TTFT_BUCKETS, 0.12, 0.45),
    ("vllm:request_queue_time_seconds", REQUEST_LATENCY_BUCKETS, 0.35, 0.9),
    ("vllm:request_prefill_time_seconds", REQUEST_LATENCY_BUCKETS, 0.4, 0.85),
    ("vllm:request_decode_time_seconds", REQUEST_LATENCY_BUCKETS, 1.6, 4.5),
    ("vllm:inter_token_latency_seconds", ITL_BUCKETS, 0.028, 0.06),
    ("vllm:e2e_request_latency_seconds", REQUEST_LATENCY_BUCKETS, 2.2, 5.5),
    ("vllm:request_prompt_tokens", None, 256.0, 900.0),  # 1-2-5 buckets
    ("vllm:request_generation_tokens", None, 128.0, 400.0),
)

GAUGES = (
    ("vllm:num_requests_running", "Number of requests currently running."),
    ("vllm:num_requests_waiting", "Number of requests waiting to be processed."),
    ("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage."),
)

COUNTERS = (
    ("vllm:prompt_tokens_total", "Number of prefill tokens processed."),
    ("vllm:generation_tokens_total", "Number of generation tokens processed."),
    ("vllm:request_success_total", "Count of successfully processed requests."),
)

MODEL_NAME = "mock-opt-125m"
LABELS = f'model_name="{MODEL_NAME}"'
AVG_PROMPT_TOKENS = 256.0
AVG_GEN_TOKENS = 128.0
DEFAULT_QPS = 2.5
# Backdate so the first scrape already has history; values still grow.
WARMUP_SECONDS = 600.0


def build_1_2_5_buckets(max_value: int) -> list[float]:
    """Match vLLM ``build_1_2_5_buckets(max_model_len)`` for token histograms."""
    buckets: list[float] = []
    mantissa = 1
    exponent = 0
    while True:
        value = mantissa * (10**exponent)
        if value > max_value:
            break
        buckets.append(float(value))
        if mantissa == 1:
            mantissa = 2
        elif mantissa == 2:
            mantissa = 5
        else:
            mantissa = 1
            exponent += 1
    return buckets


TOKEN_BUCKETS = build_1_2_5_buckets(8192)


def _le_label(value: float) -> str:
    """Prometheus ``le`` label; keep ``1.0`` so Telegraf field ``_<le>`` is numeric."""
    if math.isinf(value):
        return "+Inf"
    text = repr(float(value))
    if text.endswith(".0") and abs(value) >= 1:
        # Keep the trailing .0 for 1.0 / 10.0 - dashboard regex is [0-9.]+.
        return text
    return format(float(value), "g")


def _lognormal_params(median: float, p95: float) -> tuple[float, float]:
    mu = math.log(max(median, 1e-9))
    sigma = (math.log(max(p95, median * 1.01)) - mu) / 1.6448536269514722
    return mu, max(sigma, 0.05)


def _lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    return 0.5 * (1.0 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2.0))))


def _lognormal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + 0.5 * sigma * sigma)


class VllmMockState:
    """Wall-clock monotonic counters + histogram CDFs. Safe to snapshot often."""

    def __init__(self, qps: float = DEFAULT_QPS, now: float | None = None) -> None:
        self.qps = qps
        self.start = (now if now is not None else time.time()) - WARMUP_SECONDS
        self._lock = threading.Lock()

    def snapshot(self, now: float | None = None) -> dict:
        ts = now if now is not None else time.time()
        with self._lock:
            elapsed = max(0.0, ts - self.start)
        completed = max(1, int(elapsed * self.qps))
        wave = math.sin(elapsed / 20.0)
        running = max(1, int(round(5 + 2 * wave)))
        waiting = max(1, int(round(3 + 2 * math.sin(elapsed / 27.0))))
        kv = 0.42 + 0.12 * math.sin(elapsed / 33.0)
        prompt_tokens = int(completed * AVG_PROMPT_TOKENS)
        gen_tokens = int(completed * AVG_GEN_TOKENS)

        histograms = {}
        for name, buckets, median, p95 in HISTOGRAMS:
            bounds = TOKEN_BUCKETS if buckets is None else buckets
            mu, sigma = _lognormal_params(median, p95)
            cumul: list[tuple[str, int]] = []
            for bound in bounds:
                fraction = min(1.0, _lognormal_cdf(bound, mu, sigma))
                cumul.append((_le_label(bound), int(completed * fraction)))
            # Last finite bucket may be < count; +Inf == count (not emitted as _+Inf).
            count = completed
            hist_sum = completed * _lognormal_mean(mu, sigma)
            histograms[name] = {
                "buckets": cumul,
                "count": count,
                "sum": hist_sum,
            }

        return {
            "elapsed": elapsed,
            "completed": completed,
            "gauges": {
                "vllm:num_requests_running": float(running),
                "vllm:num_requests_waiting": float(waiting),
                "vllm:kv_cache_usage_perc": float(kv),
            },
            "counters": {
                "vllm:prompt_tokens_total": prompt_tokens,
                "vllm:generation_tokens_total": gen_tokens,
                "vllm:request_success_total": completed,
            },
            "histograms": histograms,
        }


_STATE = VllmMockState()


def _prom_labels(extra: str | None = None) -> str:
    if extra:
        return "{" + LABELS + "," + extra + "}"
    return "{" + LABELS + "}"


def render_prometheus(snap: dict | None = None) -> str:
    """Prometheus text exposition (what Telegraf scrapes from /metrics)."""
    data = snap if snap is not None else _STATE.snapshot()
    lines: list[str] = [
        "# BK-Lite local vLLM mock - V1 vllm: prefix, no GPU.",
    ]
    for name, help_text in GAUGES:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{_prom_labels()} {data['gauges'][name]}")
        lines.append("")

    for name, help_text in COUNTERS:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        extra = 'finished_reason="stop"' if name == "vllm:request_success_total" else None
        lines.append(f"{name}{_prom_labels(extra)} {data['counters'][name]}")
        lines.append("")

    for name, _buckets, _median, _p95 in HISTOGRAMS:
        hist = data["histograms"][name]
        lines.append(f"# HELP {name} Histogram family for the vLLM SLO dashboard.")
        lines.append(f"# TYPE {name} histogram")
        for le, value in hist["buckets"]:
            le_label = 'le="{}"'.format(le)
            lines.append("{}{} {}".format(name + "_bucket", _prom_labels(le_label), value))
        inf_label = 'le="+Inf"'
        lines.append("{}{} {}".format(name + "_bucket", _prom_labels(inf_label), hist["count"]))
        lines.append(f"{name}_sum{_prom_labels()} {hist['sum']:.6f}")
        lines.append(f"{name}_count{_prom_labels()} {hist['count']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def iter_telegraf_named_series(
    instance_id: str = "vllm-8000",
    snap: dict | None = None,
) -> Iterable[tuple[str, float, dict[str, str]]]:
    """Post-Telegraf VictoriaMetrics names (direct VM seeders / mock_metrics.py).

    Yields ``(metric_name, value, tags)``. Tags always include ``instance_id``.
    Histogram +Inf is omitted; dashboard uses ``_count``. Call every write
    cycle so counters and ``_count``/``_sum``/``_<le>`` keep increasing.
    """
    data = snap if snap is not None else _STATE.snapshot()
    tags = {
        "instance_id": instance_id,
        "instance_type": "vllm",
        "collect_type": "bkpull",
        "config_type": "vllm",
        "model_name": MODEL_NAME,
    }
    for name, value in data["gauges"].items():
        yield f"{name}_gauge", float(value), tags
    for name, value in data["counters"].items():
        extra = dict(tags)
        if name == "vllm:request_success_total":
            extra = dict(tags, finished_reason="stop")
        yield f"{name}_counter", float(value), extra
    for name, hist in data["histograms"].items():
        yield f"{name}_count", float(hist["count"]), tags
        yield f"{name}_sum", float(hist["sum"]), tags
        for le, value in hist["buckets"]:
            yield f"{name}_{le}", float(value), tags


def render_telegraf_dump(instance_id: str = "vllm-8000", snap: dict | None = None) -> str:
    lines = []
    for name, value, tags in iter_telegraf_named_series(instance_id, snap):
        tag_s = ",".join(f"{k}={v}" for k, v in tags.items())
        lines.append(f"{name}{{{tag_s}}} {value}")
    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        sys.stderr.write("{} - {}\n".format(self.address_string(), fmt % args))

    def _write(self, body: bytes, content_type: str = "text/plain; version=0.0.4; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/metrics", "/", "/metrics/"):
            body = render_prometheus().encode("utf-8")
            self._write(body)
            return
        if path in ("/health", "/healthz"):
            self._write(b"ok\n", "text/plain; charset=utf-8")
            return
        self.send_error(404, "use /metrics")


def serve(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), MetricsHandler)
    print(
        f"vLLM mock listening on http://{host}:{port}/metrics " f"(qps≈{_STATE.qps}, warmup={int(WARMUP_SECONDS)}s)",
        file=sys.stderr,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        httpd.server_close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local vLLM V1 Prometheus mock for bk-lite SLO dashboard.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    parser.add_argument("--qps", type=float, default=DEFAULT_QPS, help="Synthetic completed-request rate")
    parser.add_argument("--dump", action="store_true", help="Print Prometheus text once and exit")
    parser.add_argument(
        "--dump-telegraf",
        action="store_true",
        help="Print post-Telegraf VictoriaMetrics names once and exit",
    )
    parser.add_argument(
        "--instance-id",
        default="vllm-8000",
        help="instance_id tag for --dump-telegraf / iter_telegraf_named_series",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    global _STATE
    _STATE = VllmMockState(qps=args.qps)
    if args.dump:
        sys.stdout.write(render_prometheus())
        return 0
    if args.dump_telegraf:
        sys.stdout.write(render_telegraf_dump(args.instance_id))
        return 0
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
