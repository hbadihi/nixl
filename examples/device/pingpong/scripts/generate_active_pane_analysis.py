#!/usr/bin/env python3
"""Generate notebook, CSVs, and SVG charts for active-pane ping-pong sweeps.

The script intentionally uses only the Python standard library. It parses the
raw send/recv .out files as the timing source of truth. Newer benchmark runs
emit tagged ``[pingpong-stats]`` and percentile-rich ``[proxy-worker-stats]``
records; older table-style runs are retained as a parser fallback.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


OLD_METRIC_RE = re.compile(
    r"^\s*(issue->deq|post-submit|one-way|issue|prepare|submit|rtt)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
)
PINGPONG_PREFIX_RE = re.compile(r"^\[pingpong-stats\]\s+(?P<phase>\S+)\s+(?P<body>.*)$")
PINGPONG_META_RE = re.compile(r"^\[pingpong-stats\]\s+meta\s+(?P<body>.*)$")
ROLE_NAME_RE = re.compile(
    r"^(?P<prefix>send|recv)_(?P<op>.+)_(?P<level>thread|warp)_"
    r"(?P<measure>[^_]+)_stats-(?P<stats>[^_]+)_"
    r"(?P<variant>direct|proxy)_(?P<msg_size>\d+)\.out$"
)
ERR_NAME_RE = re.compile(
    r"^(?P<prefix>send|recv)_(?P<op>.+)_(?P<level>thread|warp)_"
    r"(?P<measure>[^_]+)_stats-(?P<stats>[^_]+)_"
    r"(?P<variant>direct|proxy)_(?P<msg_size>\d+)\.err$"
)
WORKER_RE = re.compile(
    r"^\[proxy-worker-stats\]\[w(?P<worker>\d+)\]\s+(?P<stage>\S+)\s+"
    r"(?P<body>.*)$"
)
WORKER_HIST_RE = re.compile(
    r"^\[proxy-worker-stats\]\[w(?P<worker>\d+)\]\s+(?P<stage>\S+)\s+hist_us=(?P<hist>.*)$"
)
POLLS_RE = re.compile(
    r"^\[proxy-worker-stats\]\[w(?P<worker>\d+)\]\s+"
    r"polls/request=\s*(?P<polls>[0-9.]+)\s+"
    r"progress_calls=(?P<progress_calls>\d+)\s+runOnce_iters=(?P<run_once_iters>\d+)"
)
NUMBER_RE = re.compile(r"(?P<name>avg|p50|p90|p99|min|max|stddev)=\s*(?P<value>[-+0-9.eE]+)\s*us")
COUNT_RE = re.compile(r"\bn=\s*(?P<count>\d+)")
META_KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_-]+)=(?P<value>\S+)")
HIST_BUCKET_RE = re.compile(r"(?P<label><0\.1|<0\.5|<1|<2|<5|<10|<50|<100|>=100):(?P<count>\d+)")

GPU_PHASE_ORDER = ["issue", "complete", "peer_wait"]
LEGACY_GPU_PHASE_ORDER = ["issue", "issue_to_deq", "prepare", "submit", "post_submit"]
SUMMARY_PHASES = ["one_way", "rtt"]
WORKER_STAGE_ORDER = [
    "dequeue",
    "prepare",
    "submit",
    "post_progress",
    "post_check",
    "post_wait",
    "publish",
]
PHASE_LABELS = {
    "issue": "gpu issue",
    "complete": "gpu complete",
    "peer_wait": "gpu peer-wait",
    "one_way": "one-way",
    "rtt": "rtt",
    "issue_to_deq": "legacy gpu issue->deq",
    "prepare": "legacy gpu prepare",
    "submit": "legacy gpu submit",
    "post_submit": "legacy gpu post-submit",
}
WORKER_LABELS = {
    "dequeue": "worker dequeue",
    "prepare": "worker prepareSubmission",
    "submit": "worker backend submit",
    "post_submit": "worker post_submit",
    "progress": "worker progress",
    "check": "worker check",
    "post_progress": "worker post_progress",
    "post_check": "worker post_check",
    "post_wait": "worker post_wait",
    "publish": "worker publish",
}
HIST_FIELD_NAMES = {
    "<0.1": "hist_lt_0_1",
    "<0.5": "hist_lt_0_5",
    "<1": "hist_lt_1",
    "<2": "hist_lt_2",
    "<5": "hist_lt_5",
    "<10": "hist_lt_10",
    "<50": "hist_lt_50",
    "<100": "hist_lt_100",
    ">=100": "hist_ge_100",
}
# Bucket order matches kHistLabels in bench_main.cpp and proxy_worker.cpp. The buckets are
# exclusive (each sample lands in the first `us < upper_bound` bucket), so the per-phase
# counts sum to the number of samples kept by the sampler.
HIST_BUCKET_ORDER = (
    ("<0.1", "hist_lt_0_1"),
    ("<0.5", "hist_lt_0_5"),
    ("<1", "hist_lt_1"),
    ("<2", "hist_lt_2"),
    ("<5", "hist_lt_5"),
    ("<10", "hist_lt_10"),
    ("<50", "hist_lt_50"),
    ("<100", "hist_lt_100"),
    (">=100", "hist_ge_100"),
)
HIST_BUCKET_COLORS = (
    "#2C7BB6",
    "#5BA1CC",
    "#8AC7E0",
    "#B6DAE7",
    "#FFFFBF",
    "#FCC480",
    "#F4A37A",
    "#D86A56",
    "#B2182B",
)
PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#BAB0AC",
    "#9D755D",
]
STAGE_COLORS = {
    "gpu issue": "#4C78A8",
    "gpu complete": "#F58518",
    "gpu peer-wait": "#54A24B",
    "legacy gpu issue->deq": "#F58518",
    "legacy gpu prepare": "#8CD17D",
    "legacy gpu submit": "#B279A2",
    "legacy gpu post-submit": "#E45756",
    "rtt (no breakdown)": "#79A7D1",
    "direct rtt": "#9B59B6",
    "direct tail": "#BAB0AC",
    "rtt unaccounted": "#D8D2C4",
    "sender peer-wait residual": "#F2C6C2",
    "worker post_submit residual": "#C7C7C7",
    "worker dequeue": "#F2B701",
    "worker prepareSubmission": "#B8E3B2",
    "worker backend submit": "#C8B4E8",
    "worker post_submit": "#F4A3A3",
    "worker progress": "#9ECAE9",
    "worker check": "#FCBBA1",
    "worker post_progress": "#A1D99B",
    "worker post_check": "#E45756",
    "worker post_wait": "#B23A48",
    "worker publish": "#777777",
}


def norm_stage(stage: str) -> str:
    return stage.replace("->", "_to_").replace("-", "_")


def denorm_stage(stage: str) -> str:
    return stage.replace("_to_", "->").replace("_", "-")


def parse_meta_body(body: str) -> dict[str, str]:
    return {match.group("key"): match.group("value") for match in META_KV_RE.finditer(body)}


def parse_stat_body(body: str) -> dict[str, Any] | None:
    count_match = COUNT_RE.search(body)
    if not count_match:
        return None
    row: dict[str, Any] = {"n": int(count_match.group("count"))}
    for match in NUMBER_RE.finditer(body):
        row[f"{match.group('name')}_us"] = float(match.group("value"))
    return row


def parse_histogram(hist: str) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for match in HIST_BUCKET_RE.finditer(hist):
        buckets[HIST_FIELD_NAMES[match.group("label")]] = int(match.group("count"))
    return buckets


STAT = "avg"
STAT_CHOICES = ("avg", "p50", "p90", "p99", "min", "max")


def phase_value(row: dict[str, Any], phase: str, stat: str | None = None) -> float:
    use_stat = stat or STAT
    value = row.get(f"{phase}_{use_stat}_us")
    if value is None and use_stat != "avg":
        value = row.get(f"{phase}_avg_us")
    if value is None:
        return 0.0
    return float(value)


def row_metric(row: dict[str, Any], key: str, stat: str | None = None) -> float | None:
    use_stat = stat or STAT
    value = row.get(f"{key}_{use_stat}_us")
    if value is None and use_stat != "avg":
        value = row.get(f"{key}_avg_us")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_new_gpu_phases(row: dict[str, Any]) -> bool:
    return any(f"{phase}_avg_us" in row for phase in ("complete", "peer_wait"))


def esc(value: Any) -> str:
    return html.escape(str(value))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_summary(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "summary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_script_run_index(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "runs.csv"
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            role = (row.get("role") or "").strip()
            if role not in ("sender", "receiver"):
                continue
            size_str = (row.get("msg_size") or "").strip()
            try:
                int(size_str)
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def parse_role_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*_*.out")):
        match = ROLE_NAME_RE.match(path.name)
        if not match:
            continue
        text = path.read_text(errors="replace")
        if "metrics:" not in text and "[pingpong-stats]" not in text:
            continue
        meta = match.groupdict()
        row: dict[str, Any] = {
            "role": "sender" if meta["prefix"] == "send" else "receiver",
            "op": meta["op"],
            "level": meta["level"],
            "measure": meta["measure"],
            "proxy_stats": meta["stats"],
            "variant": meta["variant"],
            "msg_size": int(meta["msg_size"]),
            "file": path.name,
        }
        phase_hists: dict[str, dict[str, int]] = defaultdict(dict)
        saw_pingpong_stats = False
        for line in text.splitlines():
            if "samples=" in line:
                sample_match = re.search(r"samples=(\d+)", line)
                iter_match = re.search(r"iters=(\d+)", line)
                if sample_match:
                    row["samples"] = int(sample_match.group(1))
                if iter_match:
                    row["iters"] = int(iter_match.group(1))
            meta_match = PINGPONG_META_RE.match(line)
            if meta_match:
                saw_pingpong_stats = True
                tagged_meta = parse_meta_body(meta_match.group("body"))
                if tagged_meta.get("iters"):
                    row["iters"] = int(tagged_meta["iters"])
                if tagged_meta.get("level"):
                    row["reported_level"] = tagged_meta["level"].lower()
                continue

            ping_match = PINGPONG_PREFIX_RE.match(line)
            if ping_match and ping_match.group("phase") != "meta":
                saw_pingpong_stats = True
                phase = norm_stage(ping_match.group("phase"))
                body = ping_match.group("body")
                if "hist_us=" in body:
                    phase_hists[phase].update(parse_histogram(body.split("hist_us=", 1)[1]))
                    continue
                parsed = parse_stat_body(body)
                if parsed:
                    for key, value in parsed.items():
                        suffix = "count" if key == "n" else key
                        row[f"{phase}_{suffix}"] = value
                continue

            metric_match = OLD_METRIC_RE.match(line)
            if metric_match:
                stage = norm_stage(metric_match.group(1))
                row[f"{stage}_avg_us"] = float(metric_match.group(2))
                row[f"{stage}_min_us"] = float(metric_match.group(3))
                row[f"{stage}_max_us"] = float(metric_match.group(4))
                row[f"{stage}_stddev_us"] = float(metric_match.group(5))
        for phase, buckets in phase_hists.items():
            for hist_name, count in buckets.items():
                row[f"{phase}_{hist_name}"] = count
        row["source_format"] = "pingpong-stats" if saw_pingpong_stats else "legacy-table"
        rows.append(row)
    return rows


def parse_worker_stats(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*_*.err")):
        match = ERR_NAME_RE.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        current: dict[tuple[int, str], dict[str, Any]] = {}
        poll_info_by_worker: dict[int, dict[str, str]] = {}
        for line in path.read_text(errors="replace").splitlines():
            hist_match = WORKER_HIST_RE.match(line)
            if hist_match:
                worker = int(hist_match.group("worker"))
                stage = norm_stage(hist_match.group("stage"))
                current.setdefault(
                    (worker, stage),
                    {
                        "role": "sender" if meta["prefix"] == "send" else "receiver",
                        "op": meta["op"],
                        "level": meta["level"],
                        "measure": meta["measure"],
                        "proxy_stats": meta["stats"],
                        "variant": meta["variant"],
                        "msg_size": int(meta["msg_size"]),
                        "worker": worker,
                        "stage": stage,
                        "file": path.name,
                    },
                ).update(parse_histogram(hist_match.group("hist")))
                continue

            worker_match = WORKER_RE.match(line)
            if worker_match:
                data = worker_match.groupdict()
                if data["stage"] == "polls/request":
                    continue
                parsed = parse_stat_body(data["body"])
                if not parsed:
                    continue
                worker = int(data["worker"])
                stage = norm_stage(data["stage"])
                row = current.setdefault(
                    (worker, stage),
                    {
                        "role": "sender" if meta["prefix"] == "send" else "receiver",
                        "op": meta["op"],
                        "level": meta["level"],
                        "measure": meta["measure"],
                        "proxy_stats": meta["stats"],
                        "variant": meta["variant"],
                        "msg_size": int(meta["msg_size"]),
                        "worker": worker,
                        "stage": stage,
                        "file": path.name,
                    },
                )
                for key, value in parsed.items():
                    row["n" if key == "n" else key] = value
            poll_match = POLLS_RE.match(line)
            if poll_match:
                poll_info = poll_match.groupdict()
                poll_info_by_worker[int(poll_info["worker"])] = poll_info
        for row in current.values():
            poll_info = poll_info_by_worker.get(row["worker"])
            if poll_info:
                row["polls_per_request"] = float(poll_info["polls"])
                row["progress_calls"] = int(poll_info["progress_calls"])
                row["run_once_iters"] = int(poll_info["run_once_iters"])
        rows.extend(current.values())
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phases = ["issue", "complete", "peer_wait", "issue_to_deq", "prepare", "submit", "post_submit"] + SUMMARY_PHASES
    suffixes = [
        "count",
        "avg_us",
        "p50_us",
        "p90_us",
        "p99_us",
        "min_us",
        "max_us",
        "stddev_us",
        *HIST_FIELD_NAMES.values(),
    ]
    out: list[dict[str, Any]] = []
    for row in rows:
        base = {
            key: row.get(key)
            for key in [
                "role",
                "op",
                "level",
                "measure",
                "proxy_stats",
                "variant",
                "msg_size",
                "file",
                "iters",
                "samples",
                "source_format",
            ]
            if key in row
        }
        for phase in phases:
            phase_values = {}
            for suffix in suffixes:
                key = f"{phase}_{suffix}"
                if key in row:
                    phase_values["n" if suffix == "count" else suffix] = row[key]
            if not phase_values:
                continue
            metric_row = dict(base)
            metric_row["phase"] = denorm_stage(phase)
            metric_row.update(phase_values)
            out.append(metric_row)
    return out


def derive_runs(run_dir: Path, metric_rows: list[dict[str, Any]], script_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    script_by_key = {
        (
            row.get("role"),
            row.get("op"),
            row.get("level"),
            row.get("measure"),
            row.get("proxy_stats"),
            row.get("variant"),
            int(row.get("msg_size", 0) or 0),
        ): row
        for row in script_rows
        if row.get("msg_size")
    }
    keys = sorted(
        {
            (
                row["op"],
                row["level"],
                row["measure"],
                row["proxy_stats"],
                row["variant"],
                row["msg_size"],
            )
            for row in metric_rows
        },
        key=lambda key: (key[0], key[1], key[2], key[3], key[4], key[5]),
    )
    out: list[dict[str, Any]] = []
    for op, level, measure, proxy_stats, variant, msg_size in keys:
        row: dict[str, Any] = {
            "op": op,
            "level": level,
            "measure": measure,
            "proxy_stats": proxy_stats,
            "variant": variant,
            "msg_size": msg_size,
            "sender_out": f"send_{op}_{level}_{measure}_stats-{proxy_stats}_{variant}_{msg_size}.out",
            "sender_err": f"send_{op}_{level}_{measure}_stats-{proxy_stats}_{variant}_{msg_size}.err",
            "receiver_out": f"recv_{op}_{level}_{measure}_stats-{proxy_stats}_{variant}_{msg_size}.out",
            "receiver_err": f"recv_{op}_{level}_{measure}_stats-{proxy_stats}_{variant}_{msg_size}.err",
        }
        for prefix in ("sender", "receiver"):
            out_name = row[f"{prefix}_out"]
            err_name = row[f"{prefix}_err"]
            row[f"{prefix}_out_exists"] = (run_dir / out_name).exists()
            row[f"{prefix}_err_exists"] = (run_dir / err_name).exists()
            script = script_by_key.get((prefix, op, level, measure, proxy_stats, variant, msg_size))
            if script:
                row[f"{prefix}_exit_status"] = script.get("exit_status", "")
                row[f"{prefix}_listen_port"] = script.get("listen_port", "")
                row[f"{prefix}_peer_port"] = script.get("peer_port", "")
        out.append(row)
    return out


def packet_on_wire_components(
    row: dict[str, Any],
    worker_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Return the per-component breakdown that makes up `packet_on_wire_us`.

    Direct rows: only `gpu_issue_us` is populated. Proxy stats-on rows add
    `worker_dequeue_us`, `worker_prepare_us`, `worker_submit_us`. The total in
    `packet_on_wire_us` is the sum of the populated components. Returns `None`
    for the total when components are missing (e.g. proxy stats-off, which has
    no worker stats).
    """

    if not has_new_gpu_phases(row):
        return {
            "gpu_issue_us": None,
            "worker_dequeue_us": None,
            "worker_prepare_us": None,
            "worker_submit_us": None,
            "packet_on_wire_us": None,
        }
    issue = phase_value(row, "issue")
    components: dict[str, float | None] = {
        "gpu_issue_us": issue,
        "worker_dequeue_us": None,
        "worker_prepare_us": None,
        "worker_submit_us": None,
        "packet_on_wire_us": None,
    }
    if row["variant"] == "direct":
        components["packet_on_wire_us"] = issue
        return components
    if row.get("proxy_stats") != "on":
        return components
    dequeue = worker_value(
        worker_rows, row["role"], row["op"], row["level"], row["proxy_stats"], row["msg_size"], "dequeue"
    )
    prepare = worker_value(
        worker_rows, row["role"], row["op"], row["level"], row["proxy_stats"], row["msg_size"], "prepare"
    )
    submit = worker_value(
        worker_rows, row["role"], row["op"], row["level"], row["proxy_stats"], row["msg_size"], "submit"
    )
    if dequeue == 0 and prepare == 0 and submit == 0:
        return components
    components["worker_dequeue_us"] = dequeue
    components["worker_prepare_us"] = prepare
    components["worker_submit_us"] = submit
    components["packet_on_wire_us"] = issue + dequeue + prepare + submit
    return components


def packet_on_wire_us(row: dict[str, Any], worker_rows: list[dict[str, Any]]) -> float | None:
    return packet_on_wire_components(row, worker_rows).get("packet_on_wire_us")


def build_packet_on_wire_rows(
    rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        components = packet_on_wire_components(row, worker_rows)
        record = {
            "role": row.get("role"),
            "op": row.get("op"),
            "level": row.get("level"),
            "measure": row.get("measure"),
            "variant": row.get("variant"),
            "proxy_stats": row.get("proxy_stats"),
            "msg_size": row.get("msg_size"),
            **components,
        }
        out.append(record)
    role_order = {"sender": 0, "receiver": 1}
    op_order = {"put": 0, "atomic-flag": 1}
    variant_order = {"direct": 0, "proxy": 1}
    stats_order = {"na": 0, "off": 1, "on": 2}
    out.sort(
        key=lambda r: (
            r.get("msg_size") or 0,
            role_order.get(r.get("role"), 99),
            op_order.get(r.get("op"), 99),
            0 if r.get("level") == "thread" else 1,
            variant_order.get(r.get("variant"), 99),
            stats_order.get(r.get("proxy_stats"), 99),
        )
    )
    return out


def derive_overhead(
    rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sender = [row for row in rows if row["role"] == "sender"]
    direct_by_key = {
        (row["op"], row["level"], row["measure"], row["msg_size"]): row
        for row in sender
        if row["variant"] == "direct" and "one_way_avg_us" in row
    }
    worker_rows = worker_rows or []
    out: list[dict[str, Any]] = []
    for row in sender:
        if row["variant"] != "proxy" or "one_way_avg_us" not in row:
            continue
        direct = direct_by_key.get((row["op"], row["level"], row["measure"], row["msg_size"]))
        if not direct:
            continue
        record = {
            "op": row["op"],
            "level": row["level"],
            "measure": row["measure"],
            "msg_size": row["msg_size"],
            "proxy_stats": row["proxy_stats"],
            "direct_one_way_us": direct["one_way_avg_us"],
            "proxy_one_way_us": row["one_way_avg_us"],
            "one_way_delta_us": row["one_way_avg_us"] - direct["one_way_avg_us"],
            "one_way_overhead_pct": 100.0
            * (row["one_way_avg_us"] / direct["one_way_avg_us"] - 1.0),
            "direct_rtt_us": direct["rtt_avg_us"],
            "proxy_rtt_us": row["rtt_avg_us"],
            "rtt_delta_us": row["rtt_avg_us"] - direct["rtt_avg_us"],
            "rtt_overhead_pct": 100.0 * (row["rtt_avg_us"] / direct["rtt_avg_us"] - 1.0),
        }
        direct_wire = packet_on_wire_us(direct, worker_rows)
        proxy_wire = packet_on_wire_us(row, worker_rows)
        if direct_wire is not None and proxy_wire is not None:
            record["direct_packet_on_wire_us"] = direct_wire
            record["proxy_packet_on_wire_us"] = proxy_wire
            record["packet_on_wire_delta_us"] = proxy_wire - direct_wire
            record["packet_on_wire_overhead_pct"] = (
                100.0 * (proxy_wire / direct_wire - 1.0) if direct_wire > 0 else None
            )
        out.append(record)
    return out


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    frac = value / exp
    if frac <= 1.5:
        step = 1.5
    elif frac <= 2:
        step = 2
    elif frac <= 5:
        step = 5
    else:
        step = 10
    return step * exp


def log_x(value: float, min_x: float, max_x: float, width: float) -> float:
    if min_x == max_x:
        return width / 2
    lo = math.log10(min_x)
    hi = math.log10(max_x)
    return (math.log10(value) - lo) / (hi - lo) * width


def line_chart_svg(
    title: str,
    y_label: str,
    series: list[dict[str, Any]],
    width: int = 1000,
    height: int = 470,
    suffix: str = "",
) -> str:
    ml, mr, mt, mb = 88, 34, 58, 88
    plot_w, plot_h = width - ml - mr, height - mt - mb
    xs = sorted({x for item in series for x in item["x"]})
    if not xs:
        return ""
    min_x, max_x = min(xs), max(xs)
    max_y = nice_max(max(max(item["y"]) for item in series if item["y"]) * 1.12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}"
        ".title{font-size:18px;font-weight:700}.axis{stroke:#333}.grid{stroke:#ddd}"
        ".label{fill:#444}.legend{font-size:12px}</style>",
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>',
    ]
    for i in range(6):
        val = max_y * i / 5
        y = mt + plot_h - (val / max_y) * plot_h
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" class="label">{val:.1f}{suffix}</text>'
        )
    for xval in xs:
        x = ml + log_x(xval, min_x, max_x, plot_w)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+plot_h}" class="grid"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{mt+plot_h+20}" text-anchor="middle" class="label">{xval:g}</text>'
        )
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" class="axis"/>')
    parts.append(
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" class="axis"/>'
    )
    for idx, item in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        points = []
        for xval, yval in zip(item["x"], item["y"]):
            x = ml + log_x(xval, min_x, max_x, plot_w)
            y = mt + plot_h - (yval / max_y) * plot_h
            points.append((x, y, xval, yval))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
        for x, y, _, yval in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
            if len(points) <= 7:
                parts.append(
                    f'<text x="{x:.1f}" y="{y-7:.1f}" text-anchor="middle" class="label">{yval:.2f}</text>'
                )
    lx, ly = ml, 40
    for idx, item in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        col = idx % 3
        row = idx // 3
        x = lx + col * 220
        y = ly + row * 18
        parts.append(f'<rect x="{x}" y="{y}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x+18}" y="{y+10}" class="legend">{esc(item["name"])}</text>')
    parts.append(f'<text x="{ml+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">Message size (bytes, log scale)</text>')
    parts.append(
        f'<text transform="translate(18 {mt+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{esc(y_label)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def grouped_bar_svg(
    title: str,
    y_label: str,
    categories: list[str],
    series: list[dict[str, Any]],
    width: int = 1180,
    height: int = 520,
    suffix: str = "",
) -> str:
    ml, mr, mt, mb = 82, 30, 62, 110
    plot_w, plot_h = width - ml - mr, height - mt - mb
    max_y = nice_max(max(max(item["values"]) for item in series if item["values"]) * 1.12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}"
        ".title{font-size:18px;font-weight:700}.axis{stroke:#333}.grid{stroke:#ddd}"
        ".label{fill:#444}.legend{font-size:12px}</style>",
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>',
    ]
    for i in range(6):
        val = max_y * i / 5
        y = mt + plot_h - (val / max_y) * plot_h
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" class="label">{val:.1f}{suffix}</text>'
        )
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" class="axis"/>')
    parts.append(
        f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" class="axis"/>'
    )
    group_w = plot_w / max(1, len(categories))
    bar_w = min(24, group_w / (len(series) + 1.4))
    for si, item in enumerate(series):
        color = PALETTE[si % len(PALETTE)]
        for ci, val in enumerate(item["values"]):
            cx = ml + group_w * ci + group_w / 2
            x = cx + (si - (len(series) - 1) / 2) * bar_w * 1.2 - bar_w / 2
            h = (val / max_y) * plot_h
            y = mt + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            if bar_w > 12:
                parts.append(
                    f'<text x="{x+bar_w/2:.1f}" y="{max(mt+12, y-4):.1f}" text-anchor="middle" class="label">{val:.2f}</text>'
                )
    for ci, category in enumerate(categories):
        cx = ml + group_w * ci + group_w / 2
        for li, line in enumerate(category.split("\n")):
            parts.append(
                f'<text x="{cx:.1f}" y="{mt+plot_h+20+li*14:.1f}" text-anchor="middle" class="label">{esc(line)}</text>'
            )
    for si, item in enumerate(series):
        color = PALETTE[si % len(PALETTE)]
        parts.append(f'<rect x="{ml}" y="{40+si*17}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{ml+18}" y="{50+si*17}" class="legend">{esc(item["name"])}</text>')
    parts.append(
        f'<text transform="translate(18 {mt+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{esc(y_label)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def row_label(row: dict[str, Any]) -> str:
    label = f"{row['role']} {row['op']} {row['level']} {row['variant']}"
    if row["variant"] == "proxy":
        label += f" stats {row['proxy_stats']}"
    return label


def stage_segments(row: dict[str, Any]) -> list[tuple[str, float]]:
    rtt = row_metric(row, "rtt")
    if has_new_gpu_phases(row):
        phases = [(PHASE_LABELS[phase], phase_value(row, phase)) for phase in GPU_PHASE_ORDER]
        phase_sum = sum(value for _, value in phases)
        if rtt is not None and row["role"] == "sender":
            phases.append(("rtt unaccounted", max(0.0, float(rtt) - phase_sum)))
        return phases

    legacy_phases = [
        (PHASE_LABELS[phase], phase_value(row, phase))
        for phase in LEGACY_GPU_PHASE_ORDER
        if f"{phase}_avg_us" in row
    ]
    if legacy_phases:
        if rtt is not None and row["role"] == "sender":
            legacy_phases.append(
                ("rtt unaccounted", max(0.0, float(rtt) - sum(value for _, value in legacy_phases)))
            )
        else:
            one_way = row_metric(row, "one_way")
            if row["variant"] == "direct" and one_way is not None:
                legacy_phases.append(
                    ("direct tail", max(0.0, float(one_way) - sum(value for _, value in legacy_phases)))
                )
        return legacy_phases

    if rtt is not None and rtt > 0 and row["role"] == "sender":
        label = "direct rtt" if row.get("variant") == "direct" else "rtt (no breakdown)"
        return [(label, float(rtt))]
    return []


def worker_value(
    worker_rows: list[dict[str, Any]],
    role: str,
    op: str,
    level: str,
    proxy_stats: str,
    msg_size: int,
    stage: str,
) -> float:
    for row in worker_rows:
        if (
            row["role"] == role
            and row["op"] == op
            and row["level"] == level
            and row["proxy_stats"] == proxy_stats
            and row["msg_size"] == msg_size
            and row["stage"] == stage
        ):
            value = row.get(f"{STAT}_us")
            if value is None and STAT != "avg":
                value = row.get("avg_us")
            try:
                return float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def worker_stage_segments(
    worker_rows: list[dict[str, Any]],
    role: str,
    op: str,
    level: str,
    proxy_stats: str,
    msg_size: int,
) -> list[tuple[str, float]]:
    # WORKER_STAGE_ORDER contains only non-overlapping per-request stages. We intentionally
    # exclude the recorded `post_submit` aggregate (since `post_submit` equals
    # `post_progress + post_check + post_wait` by construction in proxy_worker.cpp) and the
    # per-call `progress` / `check` stats (which are single-iteration durations, not
    # per-request totals — those are captured by `post_progress` / `post_check`). Summing
    # them all would double- or triple-count time on the worker lane.
    return [
        (WORKER_LABELS[stage], worker_value(worker_rows, role, op, level, proxy_stats, msg_size, stage))
        for stage in WORKER_STAGE_ORDER
    ]


def stage_segments_with_worker(row: dict[str, Any], worker_rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    if row["variant"] != "proxy":
        return stage_segments(row)
    return stage_segments(row) + worker_stage_segments(
        worker_rows, row["role"], row["op"], row["level"], row["proxy_stats"], row["msg_size"]
    )


def horizontal_path_svg(
    title: str,
    rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]] | None = None,
    width: int = 1320,
) -> str:
    if not rows:
        return ""
    gpu_h = 26
    worker_h = 18
    sub_gap = 4
    # row_gap is sized to fit the two-line label stack above each gpu lane (`1w` at
    # `gpu_y - 4` and the `complete <value>` / `packet on wire` band at `gpu_y - 16`)
    # without clipping into the previous row's bar.
    row_gap = 30

    layouts: list[dict[str, Any]] = []
    for row in rows:
        gpu_segments = [(name, value) for name, value in stage_segments(row) if value > 0]
        worker_segments: list[tuple[str, float]] = []
        if (
            worker_rows is not None
            and row.get("variant") == "proxy"
            and has_new_gpu_phases(row)
        ):
            worker_segments = [
                (name, value)
                for name, value in worker_stage_segments(
                    worker_rows, row["role"], row["op"], row["level"], row["proxy_stats"], row["msg_size"]
                )
                if value > 0
            ]
        layouts.append({"row": row, "gpu": gpu_segments, "worker": worker_segments})

    total_rows_h = 0
    for layout in layouts:
        layout["height"] = gpu_h + (sub_gap + worker_h if layout["worker"] else 0)
        total_rows_h += layout["height"]
    total_rows_h += row_gap * max(0, len(layouts) - 1)

    ml, mr, mt, mb = 270, 60, 112, 78
    height = mt + total_rows_h + mb
    plot_w = width - ml - mr

    rtt_max = max((row_metric(layout["row"], "rtt") or 0.0 for layout in layouts), default=0.0)
    gpu_max = max((sum(value for _, value in layout["gpu"]) for layout in layouts), default=0.0)
    worker_max = max((sum(value for _, value in layout["worker"]) for layout in layouts), default=0.0)
    raw_max = max(rtt_max, gpu_max, worker_max)
    max_end = nice_max(raw_max * 1.05) if raw_max > 0 else 1.0

    used_labels: list[str] = []
    seen: set[str] = set()
    for layout in layouts:
        for name, value in layout["gpu"] + layout["worker"]:
            if value > 0 and name not in seen and name in STAGE_COLORS:
                seen.add(name)
                used_labels.append(name)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}"
        ".title{font-size:18px;font-weight:700}.axis{stroke:#333}.grid{stroke:#ddd}"
        ".marker{stroke:#111;stroke-width:1.5;stroke-dasharray:4 3}.label{fill:#444}"
        ".small{font-size:11px;fill:#555}.lane{font-size:10px;fill:#666}</style>",
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>',
    ]
    for i in range(6):
        val = max_end * i / 5
        x = ml + (val / max_end) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{mt-26}" x2="{x:.1f}" y2="{height-mb+10}" class="grid"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height-mb+28}" text-anchor="middle" class="label">{val:.2f}</text>'
        )

    legend_y = 50
    legend_x = ml
    for idx, name in enumerate(used_labels):
        if idx > 0 and idx % 6 == 0:
            legend_y += 18
            legend_x = ml
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{STAGE_COLORS[name]}"/>')
        parts.append(f'<text x="{legend_x+17}" y="{legend_y+10}" class="small">{esc(name)}</text>')
        legend_x += 175 if name.startswith("worker ") else 145
    parts.append(
        f'<text x="{ml+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">'
        f'Elapsed time (us). GPU bars are sender-observed RTT timeline; worker bars are CPU-local durations on their own zero baseline.'
        f'</text>'
    )

    cursor_y = mt
    for layout in layouts:
        row = layout["row"]
        gpu_y = cursor_y
        worker_y = cursor_y + gpu_h + sub_gap
        bottom = worker_y + worker_h if layout["worker"] else gpu_y + gpu_h
        center_y = (gpu_y + bottom) / 2
        parts.append(
            f'<text x="{ml-10}" y="{center_y+4:.1f}" text-anchor="end" class="label">{esc(row_label(row))}</text>'
        )
        if layout["worker"]:
            parts.append(f'<text x="{ml-12}" y="{gpu_y+gpu_h/2+4:.1f}" text-anchor="end" class="lane">gpu</text>')
            parts.append(
                f'<text x="{ml-12}" y="{worker_y+worker_h/2+4:.1f}" text-anchor="end" class="lane">worker</text>'
            )

        worker_total = sum(value for _, value in layout["worker"])
        gpu_segment_values = dict(layout["gpu"])
        worker_segment_values = dict(layout["worker"])
        gpu_complete = gpu_segment_values.get("gpu complete", 0.0)
        if has_new_gpu_phases(row):
            gpu_issue_offset = phase_value(row, "issue")
        else:
            gpu_issue_offset = phase_value(row, "issue") + phase_value(row, "issue_to_deq")

        cursor = 0.0
        gpu_complete_end_x: float | None = None
        for name, value in layout["gpu"]:
            x = ml + (cursor / max_end) * plot_w
            w = (value / max_end) * plot_w
            parts.append(
                f'<rect x="{x:.1f}" y="{gpu_y:.1f}" width="{w:.1f}" height="{gpu_h}" fill="{STAGE_COLORS[name]}"/>'
            )
            if w > 36:
                parts.append(
                    f'<text x="{x+w/2:.1f}" y="{gpu_y+gpu_h/2+4:.1f}" text-anchor="middle" class="small">{value:.2f}</text>'
                )
            cursor += value
            if name == "gpu complete":
                gpu_complete_end_x = ml + (cursor / max_end) * plot_w

        # The GPU lane Σ is omitted because Σ_gpu == RTT by construction (rtt unaccounted
        # absorbs the difference). Instead surface the gpu complete value with a tick so
        # it remains readable when the bar is wide. The text sits on the upper label
        # band (`gpu_y - 16`) shared with `packet on wire` to avoid overlapping the `1w`
        # label on the lower band (`gpu_y - 4`).
        if gpu_complete_end_x is not None and gpu_complete > 0:
            parts.append(
                f'<line x1="{gpu_complete_end_x:.1f}" y1="{gpu_y-12}" x2="{gpu_complete_end_x:.1f}" y2="{gpu_y}" stroke="#555"/>'
            )
            parts.append(
                f'<text x="{gpu_complete_end_x-3:.1f}" y="{gpu_y-16}" text-anchor="end" class="small">complete {gpu_complete:.2f}</text>'
            )

        # Faint shading on the worker lane that spans the gpu complete interval makes the
        # temporal nesting visible without distorting any bar widths.
        if layout["worker"] and gpu_complete > 0:
            shade_x = ml + (gpu_issue_offset / max_end) * plot_w
            shade_w = (gpu_complete / max_end) * plot_w
            parts.append(
                f'<rect x="{shade_x:.1f}" y="{worker_y:.1f}" width="{shade_w:.1f}" height="{worker_h}" fill="#888" fill-opacity="0.12"/>'
            )

        cursor = gpu_issue_offset
        for name, value in layout["worker"]:
            x = ml + (cursor / max_end) * plot_w
            w = (value / max_end) * plot_w
            parts.append(
                f'<rect x="{x:.1f}" y="{worker_y:.1f}" width="{w:.1f}" height="{worker_h}" fill="{STAGE_COLORS[name]}"/>'
            )
            if w > 36:
                parts.append(
                    f'<text x="{x+w/2:.1f}" y="{worker_y+worker_h/2+4:.1f}" text-anchor="middle" class="small">{value:.2f}</text>'
                )
            cursor += value
        if worker_total > 0:
            total_x = ml + ((gpu_issue_offset + worker_total) / max_end) * plot_w
            parts.append(
                f'<text x="{total_x+6:.1f}" y="{worker_y-2:.1f}" class="small">\u03a3 {worker_total:.2f}</text>'
            )

        worker_dequeue = worker_segment_values.get("worker dequeue", 0.0)
        worker_prepare = worker_segment_values.get("worker prepareSubmission", 0.0)
        worker_submit = worker_segment_values.get("worker backend submit", 0.0)
        # The direct row's analog of "packet on wire" is end-of-`gpu issue`, so the marker
        # rendering is unified across direct and proxy: anchor is `gpu_issue_offset`, and
        # proxy rows shift it by the host-side worker stages that precede the wire.
        if has_new_gpu_phases(row):
            if layout["worker"]:
                wire_value = (
                    gpu_issue_offset + worker_dequeue + worker_prepare + worker_submit
                )
            else:
                wire_value = gpu_issue_offset
            if wire_value > 0:
                wire_x = ml + (wire_value / max_end) * plot_w
                parts.append(
                    f'<line x1="{wire_x:.1f}" y1="{gpu_y-2}" x2="{wire_x:.1f}" y2="{bottom+2}" class="marker"/>'
                )
                # `packet on wire` shares the upper label band (`gpu_y - 16`) with the
                # `complete <value>` tick label; they are start- vs end-anchored at distinct
                # x positions (end-of-`worker submit` vs end-of-`gpu complete`) so they do
                # not collide. `1w` stays on the lower band (`gpu_y - 4`).
                parts.append(
                    f'<text x="{wire_x+4:.1f}" y="{gpu_y-16}" class="small">packet on wire {wire_value:.2f}</text>'
                )

        one_way = row_metric(row, "one_way")
        if one_way is not None:
            one_x = ml + (one_way / max_end) * plot_w
            parts.append(
                f'<line x1="{one_x:.1f}" y1="{gpu_y-2}" x2="{one_x:.1f}" y2="{bottom+2}" class="marker"/>'
            )
            parts.append(
                f'<text x="{one_x+4:.1f}" y="{gpu_y-4}" class="small">1w {one_way:.2f}</text>'
            )
        rtt_value = row_metric(row, "rtt")
        if rtt_value is not None:
            rtt_x = ml + (rtt_value / max_end) * plot_w
            parts.append(
                f'<line x1="{rtt_x:.1f}" y1="{gpu_y}" x2="{rtt_x:.1f}" y2="{bottom}" stroke="#111"/>'
            )
            parts.append(
                f'<text x="{rtt_x+5:.1f}" y="{gpu_y+gpu_h/2+4:.1f}" class="small">RTT {rtt_value:.2f}</text>'
            )

        cursor_y += layout["height"] + row_gap

    parts.append("</svg>")
    return "\n".join(parts)


def _read_hist_counts(source: dict[str, Any], prefix: str = "") -> tuple[list[int], int]:
    counts: list[int] = []
    for _, field in HIST_BUCKET_ORDER:
        key = f"{prefix}{field}" if prefix else field
        raw = source.get(key)
        try:
            counts.append(int(raw) if raw not in (None, "") else 0)
        except (TypeError, ValueError):
            counts.append(0)
    return counts, sum(counts)


# Columns shared by the unified phase_breakdown.csv and per-chart sidecar CSVs. Kept in
# a fixed order so external consumers can rely on the schema regardless of which rows
# happen to populate which field.
_BREAKDOWN_BASE_COLS = (
    "msg_size",
    "role",
    "op",
    "level",
    "variant",
    "proxy_stats",
    "source",
    "phase_key",
    "phase_label",
    "worker",
    "n",
    "hist_total",
    "avg_us",
    "p50_us",
    "p90_us",
    "p99_us",
    "min_us",
    "max_us",
    "stddev_us",
)
_BREAKDOWN_HIST_COLS = tuple(field for _, field in HIST_BUCKET_ORDER)
_BREAKDOWN_COLS = _BREAKDOWN_BASE_COLS + _BREAKDOWN_HIST_COLS


def _coerce_float(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _gpu_phase_breakdown_row(row: dict[str, Any], phase: str) -> dict[str, Any] | None:
    counts, hist_total = _read_hist_counts(row, f"{phase}_")
    count_field = row.get(f"{phase}_count")
    avg_field = row.get(f"{phase}_avg_us")
    if (
        hist_total <= 0
        and count_field in (None, "", 0)
        and avg_field in (None, "")
    ):
        return None
    record: dict[str, Any] = {
        "msg_size": row.get("msg_size"),
        "role": row.get("role"),
        "op": row.get("op"),
        "level": row.get("level"),
        "variant": row.get("variant"),
        "proxy_stats": row.get("proxy_stats"),
        "source": "gpu",
        "phase_key": phase,
        "phase_label": PHASE_LABELS.get(phase, denorm_stage(phase)),
        "worker": None,
        "n": _coerce_int(count_field),
        "hist_total": hist_total,
        "avg_us": _coerce_float(avg_field),
        "p50_us": _coerce_float(row.get(f"{phase}_p50_us")),
        "p90_us": _coerce_float(row.get(f"{phase}_p90_us")),
        "p99_us": _coerce_float(row.get(f"{phase}_p99_us")),
        "min_us": _coerce_float(row.get(f"{phase}_min_us")),
        "max_us": _coerce_float(row.get(f"{phase}_max_us")),
        "stddev_us": _coerce_float(row.get(f"{phase}_stddev_us")),
    }
    for (_, field), count in zip(HIST_BUCKET_ORDER, counts):
        record[field] = count
    return record


def _worker_phase_breakdown_row(worker_row: dict[str, Any]) -> dict[str, Any] | None:
    counts, hist_total = _read_hist_counts(worker_row)
    n_field = worker_row.get("n")
    avg_field = worker_row.get("avg_us")
    if hist_total <= 0 and n_field in (None, "", 0) and avg_field in (None, ""):
        return None
    stage = worker_row.get("stage", "")
    record: dict[str, Any] = {
        "msg_size": worker_row.get("msg_size"),
        "role": worker_row.get("role"),
        "op": worker_row.get("op"),
        "level": worker_row.get("level"),
        "variant": worker_row.get("variant"),
        "proxy_stats": worker_row.get("proxy_stats"),
        "source": "worker",
        "phase_key": stage,
        "phase_label": WORKER_LABELS.get(stage, f"worker {denorm_stage(stage)}"),
        "worker": _coerce_int(worker_row.get("worker")),
        "n": _coerce_int(n_field),
        "hist_total": hist_total,
        "avg_us": _coerce_float(avg_field),
        "p50_us": _coerce_float(worker_row.get("p50_us")),
        "p90_us": _coerce_float(worker_row.get("p90_us")),
        "p99_us": _coerce_float(worker_row.get("p99_us")),
        "min_us": _coerce_float(worker_row.get("min_us")),
        "max_us": _coerce_float(worker_row.get("max_us")),
        "stddev_us": _coerce_float(worker_row.get("stddev_us")),
    }
    for (_, field), count in zip(HIST_BUCKET_ORDER, counts):
        record[field] = count
    return record


def build_phase_breakdown_rows(
    metric_rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (config × phase) covering both GPU phases and worker stages.

    GPU rows cover the new pingpong-stats phases plus the legacy micro-stages and summary
    phases (one_way / rtt). Worker rows are kept at per-worker granularity so the
    aggregate ``post_submit`` identity (``post_submit == post_progress + post_check +
    post_wait``) can be verified directly from the CSV.
    """

    gpu_phase_keys = (
        list(GPU_PHASE_ORDER)
        + list(LEGACY_GPU_PHASE_ORDER)
        + list(SUMMARY_PHASES)
    )
    seen: set[str] = set()
    ordered_gpu_keys: list[str] = []
    for key in gpu_phase_keys:
        if key not in seen:
            seen.add(key)
            ordered_gpu_keys.append(key)

    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        for phase in ordered_gpu_keys:
            record = _gpu_phase_breakdown_row(row, phase)
            if record is not None:
                rows.append(record)
    for w in worker_rows:
        record = _worker_phase_breakdown_row(w)
        if record is not None:
            rows.append(record)

    role_order = {"sender": 0, "receiver": 1}
    op_order = {"put": 0, "atomic-flag": 1}
    variant_order = {"direct": 0, "proxy": 1}
    stats_order = {"na": 0, "off": 1, "on": 2}
    source_order = {"gpu": 0, "worker": 1}
    gpu_phase_rank = {phase: idx for idx, phase in enumerate(ordered_gpu_keys)}
    worker_phase_rank = {stage: idx for idx, stage in enumerate(WORKER_STAGE_ORDER)}
    rows.sort(
        key=lambda r: (
            r.get("msg_size") or 0,
            role_order.get(r.get("role"), 99),
            op_order.get(r.get("op"), 99),
            0 if r.get("level") == "thread" else 1,
            variant_order.get(r.get("variant"), 99),
            stats_order.get(r.get("proxy_stats"), 99),
            source_order.get(r.get("source"), 99),
            r.get("worker") if r.get("worker") is not None else -1,
            (
                gpu_phase_rank.get(r.get("phase_key"), 99)
                if r.get("source") == "gpu"
                else worker_phase_rank.get(r.get("phase_key"), 99)
            ),
        )
    )
    return rows


def write_phase_breakdown_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_BREAKDOWN_COLS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in _BREAKDOWN_COLS})


def _phase_histogram_sections(
    rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
    msg_size: int,
    stats_filter: str,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row["msg_size"] == msg_size
        and row["role"] == "sender"
        and (row["variant"] == "direct" or row["proxy_stats"] == stats_filter)
    ]
    candidates.sort(
        key=lambda row: (
            0 if row["op"] == "put" else 1,
            0 if row["level"] == "thread" else 1,
            0 if row["variant"] == "direct" else 1,
        )
    )
    sections: list[dict[str, Any]] = []
    for row in candidates:
        section_phases: list[dict[str, Any]] = []
        gpu_phase_order = GPU_PHASE_ORDER if has_new_gpu_phases(row) else LEGACY_GPU_PHASE_ORDER
        for phase in gpu_phase_order:
            counts, total = _read_hist_counts(row, f"{phase}_")
            if total <= 0:
                continue
            label = PHASE_LABELS.get(phase, phase)
            section_phases.append({
                "label": label,
                "color": STAGE_COLORS.get(label, "#888"),
                "counts": counts,
                "total": total,
                "avg": phase_value(row, phase),
            })
        if row.get("variant") == "proxy" and worker_rows:
            for stage in WORKER_STAGE_ORDER:
                stage_rows = [
                    w for w in worker_rows
                    if w["role"] == row["role"]
                    and w["op"] == row["op"]
                    and w["level"] == row["level"]
                    and w["proxy_stats"] == row["proxy_stats"]
                    and w["msg_size"] == row["msg_size"]
                    and w["stage"] == stage
                ]
                if not stage_rows:
                    continue
                counts = [0] * len(HIST_BUCKET_ORDER)
                total = 0
                avg_num = 0.0
                avg_den = 0.0
                for w in stage_rows:
                    wc, wt = _read_hist_counts(w)
                    if wt <= 0:
                        continue
                    counts = [a + b for a, b in zip(counts, wc)]
                    total += wt
                    n = float(w.get("n") or 0.0)
                    avg = float(w.get(f"{STAT}_us") or w.get("avg_us") or 0.0)
                    avg_num += avg * n
                    avg_den += n
                if total <= 0:
                    continue
                label = WORKER_LABELS.get(stage, stage)
                section_phases.append({
                    "label": label,
                    "color": STAGE_COLORS.get(label, "#888"),
                    "counts": counts,
                    "total": total,
                    "avg": (avg_num / avg_den) if avg_den > 0 else 0.0,
                })
        if section_phases:
            sections.append({"row": row, "phases": section_phases})
    return sections


def phase_histogram_svg(
    title: str,
    rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
    msg_size: int,
    stats_filter: str,
    width: int = 1320,
) -> str:
    sections = _phase_histogram_sections(rows, worker_rows, msg_size, stats_filter)
    if not sections:
        return ""

    ml, mr, mt, mb = 290, 240, 116, 60
    bar_h = 14
    phase_gap = 4
    section_gap = 18
    section_header_h = 22
    plot_w = width - ml - mr

    total_phase_rows = sum(len(section["phases"]) for section in sections)
    total_h = (
        section_header_h * len(sections)
        + (bar_h + phase_gap) * total_phase_rows
        + section_gap * max(0, len(sections) - 1)
    )
    height = mt + total_h + mb

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}"
        ".title{font-size:18px;font-weight:700}.section{font-size:13px;font-weight:600;fill:#222}"
        ".phase{font-size:11px;fill:#333}.tag{font-size:10px;fill:#fff}.tag-dark{font-size:10px;fill:#222}"
        ".small{font-size:11px;fill:#555}.legend{font-size:11px;fill:#333}</style>",
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>',
    ]

    legend_y = 52
    legend_x = ml
    for idx, (label, _) in enumerate(HIST_BUCKET_ORDER):
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="{HIST_BUCKET_COLORS[idx]}" stroke="#999"/>'
        )
        parts.append(f'<text x="{legend_x+19}" y="{legend_y+11}" class="legend">{esc(label)} us</text>')
        legend_x += 110
    parts.append(
        f'<text x="{ml}" y="{legend_y+32}" class="small">'
        f'Each bar is a per-phase latency histogram; segment widths are the share of samples in each bucket. '
        f'Right edge shows total samples and {STAT} average for that phase.</text>'
    )

    y = mt
    for s_idx, section in enumerate(sections):
        row = section["row"]
        header = f"{row['role']} {row['op']} {row['level']} {row['variant']}"
        if row["variant"] == "proxy":
            header += f" stats {row['proxy_stats']}"
        parts.append(
            f'<text x="{ml-10}" y="{y+15:.1f}" text-anchor="end" class="section">{esc(header)}</text>'
        )
        parts.append(
            f'<line x1="{ml}" y1="{y+18:.1f}" x2="{width-mr}" y2="{y+18:.1f}" stroke="#ddd"/>'
        )
        y += section_header_h
        for phase in section["phases"]:
            parts.append(
                f'<text x="{ml-10}" y="{y+bar_h*0.78:.1f}" text-anchor="end" class="phase">{esc(phase["label"])}</text>'
            )
            cursor = float(ml)
            total = max(1, phase["total"])
            for idx, count in enumerate(phase["counts"]):
                if count <= 0:
                    continue
                seg_w = (count / total) * plot_w
                parts.append(
                    f'<rect x="{cursor:.2f}" y="{y:.1f}" width="{seg_w:.2f}" height="{bar_h}" fill="{HIST_BUCKET_COLORS[idx]}"/>'
                )
                if seg_w >= 32:
                    pct = 100.0 * count / total
                    tag_class = "tag-dark" if idx in (3, 4) else "tag"
                    parts.append(
                        f'<text x="{cursor+seg_w/2:.2f}" y="{y+bar_h*0.78:.1f}" text-anchor="middle" class="{tag_class}">{count} ({pct:.1f}%)</text>'
                    )
                cursor += seg_w
            parts.append(
                f'<text x="{width-mr+8}" y="{y+bar_h*0.78:.1f}" class="small">n={phase["total"]} {STAT}={phase["avg"]:.2f}us</text>'
            )
            y += bar_h + phase_gap
        if s_idx + 1 < len(sections):
            y += section_gap

    parts.append("</svg>")
    return "\n".join(parts)


def four_lane_pingpong_svg(
    title: str,
    rows: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
    msg_size: int,
    stats_filter: str | None = None,
    width: int = 1420,
) -> str:
    sender_by_key = {
        (row["op"], row["level"], row["proxy_stats"]): row
        for row in rows
        if row["role"] == "sender" and row["variant"] == "proxy" and row["msg_size"] == msg_size
        and (stats_filter is None or row["proxy_stats"] == stats_filter)
    }
    receiver_by_key = {
        (row["op"], row["level"], row["proxy_stats"]): row
        for row in rows
        if row["role"] == "receiver" and row["variant"] == "proxy" and row["msg_size"] == msg_size
        and (stats_filter is None or row["proxy_stats"] == stats_filter)
    }
    keys = sorted(
        set(sender_by_key) & set(receiver_by_key),
        key=lambda key: (
            0 if key[0] == "put" else 1,
            0 if key[1] == "thread" else 1,
            0 if key[2] == "off" else 1,
        ),
    )
    if not keys:
        return ""

    lane_h = 18
    lane_gap = 5
    group_gap = 18
    group_h = 4 * lane_h + 3 * lane_gap + group_gap
    ml, mr, mt, mb = 300, 48, 96, 74
    height = mt + len(keys) * group_h + mb
    plot_w = width - ml - mr
    max_end = nice_max(max(row_metric(sender_by_key[key], "rtt") or 0.0 for key in keys) * 1.05)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}"
        ".title{font-size:18px;font-weight:700}.axis{stroke:#333}.grid{stroke:#ddd}"
        ".marker{stroke:#111;stroke-width:1.5;stroke-dasharray:4 3}.label{fill:#444}"
        ".small{font-size:11px;fill:#555}.lane{font-size:11px;fill:#333}</style>",
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{esc(title)}</text>',
    ]

    def xpos(value: float) -> float:
        return ml + (value / max_end) * plot_w

    def draw_segments(
        y: float,
        start_us: float,
        segments: list[tuple[str, float]],
        lane_height: int = lane_h,
        tiny_labels: bool = False,
    ) -> None:
        cursor = start_us
        tiny_label_idx = 0
        for name, value in segments:
            if value <= 0:
                continue
            x = xpos(cursor)
            w = (value / max_end) * plot_w
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{lane_height}" fill="{STAGE_COLORS[name]}"/>'
            )
            if w > 32:
                parts.append(
                    f'<text x="{x+w/2:.1f}" y="{y+lane_height/2+4:.1f}" text-anchor="middle" class="small">{value:.2f}</text>'
                )
            elif tiny_labels:
                tick_x = max(x + 1.0, xpos(cursor + value))
                label_y = y - 4 - (tiny_label_idx % 2) * 10
                label = name[2:] if name.startswith("w ") else name
                parts.append(
                    f'<line x1="{tick_x:.1f}" y1="{y:.1f}" x2="{tick_x:.1f}" y2="{y+lane_height:.1f}" stroke="#333" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{tick_x+3:.1f}" y="{label_y:.1f}" class="small">{esc(label)} {value:.3f}</text>'
                )
                tiny_label_idx += 1
            cursor += value

    for i in range(6):
        val = max_end * i / 5
        x = xpos(val)
        parts.append(f'<line x1="{x:.1f}" y1="{mt-26}" x2="{x:.1f}" y2="{height-mb+8}" class="grid"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height-mb+28}" text-anchor="middle" class="label">{val:.1f}</text>'
        )

    legend = [
        "gpu issue",
        "gpu complete",
        "gpu peer-wait",
        "legacy gpu issue->deq",
        "legacy gpu prepare",
        "legacy gpu submit",
        "legacy gpu post-submit",
        "rtt unaccounted",
        "worker dequeue",
        "worker prepareSubmission",
        "worker backend submit",
        "worker post_progress",
        "worker post_check",
        "worker post_wait",
        "worker publish",
    ]
    lx, ly = ml, 42
    for idx, name in enumerate(legend):
        if idx == 5:
            lx, ly = ml, 62
        elif idx == 11:
            lx, ly = ml, 82
        parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{STAGE_COLORS[name]}"/>')
        parts.append(f'<text x="{lx+17}" y="{ly+10}" class="small">{esc(name)}</text>')
        lx += 190 if name.startswith("worker ") else 132
    parts.append(
        f'<text x="{ml+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">Sender RTT timeline (us); receiver return path is right-aligned to sender RTT end. Worker lanes are CPU-local measurements.</text>'
    )

    for idx, key in enumerate(keys):
        sender = sender_by_key[key]
        receiver = receiver_by_key[key]
        group_y = mt + idx * group_h
        label = f"{key[0]} {key[1]} stats {key[2]}"
        parts.append(
            f'<text x="{ml-10}" y="{group_y+2*lane_h+lane_gap:.1f}" text-anchor="end" class="label">{esc(label)}</text>'
        )

        sender_gpu_y = group_y
        sender_worker_y = group_y + lane_h + lane_gap
        receiver_gpu_y = sender_worker_y + lane_h + lane_gap
        receiver_worker_y = receiver_gpu_y + lane_h + lane_gap

        # Worker lanes are only meaningful when the matching GPU lane exposes phase
        # breakdown (otherwise there is no anchor for the worker start position).
        show_sender_worker = has_new_gpu_phases(sender)
        show_receiver_worker = has_new_gpu_phases(receiver)

        lane_labels = [("sender GPU", sender_gpu_y)]
        if show_sender_worker:
            lane_labels.append(("sender worker", sender_worker_y))
        lane_labels.append(("receiver GPU", receiver_gpu_y))
        if show_receiver_worker:
            lane_labels.append(("receiver worker", receiver_worker_y))
        for lane_label, y in lane_labels:
            parts.append(
                f'<text x="{ml-118}" y="{y+lane_h/2+4:.1f}" text-anchor="start" class="lane">{esc(lane_label)}</text>'
            )

        sender_segments = stage_segments(sender)
        sender_worker_segments = (
            worker_stage_segments(worker_rows, "sender", key[0], key[1], key[2], msg_size)
            if show_sender_worker
            else []
        )
        receiver_segments = stage_segments(receiver)
        receiver_worker_segments = (
            worker_stage_segments(worker_rows, "receiver", key[0], key[1], key[2], msg_size)
            if show_receiver_worker
            else []
        )

        sender_rtt = row_metric(sender, "rtt") or sum(value for _, value in sender_segments)
        receiver_total = sum(value for _, value in receiver_segments)
        receiver_start = max(0.0, sender_rtt - receiver_total)
        sender_worker_start = phase_value(sender, "issue") + phase_value(sender, "issue_to_deq")
        receiver_worker_start = (
            receiver_start
            + phase_value(receiver, "issue")
            + phase_value(receiver, "issue_to_deq")
        )

        draw_segments(sender_gpu_y, 0.0, sender_segments)
        if show_sender_worker:
            draw_segments(sender_worker_y, sender_worker_start, sender_worker_segments, tiny_labels=True)
        draw_segments(receiver_gpu_y, receiver_start, receiver_segments)
        if show_receiver_worker:
            draw_segments(receiver_worker_y, receiver_worker_start, receiver_worker_segments, tiny_labels=True)

        one_way = row_metric(sender, "one_way")
        if one_way is not None:
            one_x = xpos(one_way)
            parts.append(
                f'<line x1="{one_x:.1f}" y1="{group_y-2}" x2="{one_x:.1f}" y2="{receiver_worker_y+lane_h+2}" class="marker"/>'
            )
            parts.append(f'<text x="{one_x+4:.1f}" y="{group_y-5}" class="small">1w {one_way:.2f}</text>')
        rtt_x = xpos(sender_rtt)
        parts.append(
            f'<line x1="{rtt_x:.1f}" y1="{group_y}" x2="{rtt_x:.1f}" y2="{receiver_worker_y+lane_h}" stroke="#111"/>'
        )
        parts.append(
            f'<text x="{rtt_x+5:.1f}" y="{group_y+2*lane_h:.1f}" class="small">RTT {sender_rtt:.2f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def rows_for_size(rows: list[dict[str, Any]], msg_size: int) -> list[dict[str, Any]]:
    order = {"put": 0, "atomic-flag": 1}
    role_order = {"sender": 0, "receiver": 1}
    variant_order = {"direct": 0, "proxy": 1}
    stats_order = {"na": 0, "off": 1, "on": 2}
    subset = [row for row in rows if row["msg_size"] == msg_size]
    return sorted(
        subset,
        key=lambda row: (
            order.get(row["op"], 99),
            0 if row["level"] == "thread" else 1,
            role_order.get(row["role"], 99),
            variant_order.get(row["variant"], 99),
            stats_order.get(row["proxy_stats"], 99),
        ),
    )


def generate_charts(
    out_dir: Path,
    rows: list[dict[str, Any]],
    overhead: list[dict[str, Any]],
    worker_rows: list[dict[str, Any]],
    selected_sizes: list[int] | None = None,
) -> list[str]:
    generated: list[str] = []
    sender_rows = [row for row in rows if row["role"] == "sender"]
    ops = sorted({row["op"] for row in sender_rows})
    levels = sorted({row["level"] for row in sender_rows})

    # Sweep runs sweep `--measure-submit` as a separate axis (`measure` in the run
    # filename / CSVs). Without splitting on it, every cross-size line carries two
    # y-values per x and renders as a jagged zig-zag; treat each measure mode as its
    # own series instead.
    measures = sorted({row.get("measure", "") for row in sender_rows if row.get("measure")})

    def _series_suffix(measure: str) -> str:
        return f" measure {measure}" if len(measures) > 1 else ""

    for op in ops:
        for level in levels:
            subset = [row for row in sender_rows if row["op"] == op and row["level"] == level]
            if not subset:
                continue
            series = []
            for measure in measures:
                for variant, stats, name in [
                    ("direct", "na", "direct"),
                    ("proxy", "off", "proxy stats off"),
                    ("proxy", "on", "proxy stats on"),
                ]:
                    points = []
                    for row in subset:
                        if (
                            row["variant"] != variant
                            or row["proxy_stats"] != stats
                            or row.get("measure", "") != measure
                        ):
                            continue
                        value = row_metric(row, "one_way")
                        if value is None:
                            continue
                        points.append((row["msg_size"], value))
                    points.sort()
                    if points:
                        series.append(
                            {
                                "name": f"{name}{_series_suffix(measure)}",
                                "x": [p[0] for p in points],
                                "y": [p[1] for p in points],
                            }
                        )
            if series:
                name = f"one_way_by_size_{safe_name(op)}_{level}.svg"
                (out_dir / name).write_text(
                    line_chart_svg(
                        f"Sender one-way {STAT} latency by message size: {op} {level}",
                        f"One-way {STAT} latency (us)",
                        series,
                    )
                )
                generated.append(name)

            series = []
            for measure in measures:
                for variant, stats, name in [
                    ("direct", "na", "direct (gpu issue)"),
                    (
                        "proxy",
                        "on",
                        "proxy stats on (gpu issue + dequeue + prepare + submit)",
                    ),
                ]:
                    points = []
                    for row in subset:
                        if (
                            row["variant"] != variant
                            or row["proxy_stats"] != stats
                            or row.get("measure", "") != measure
                        ):
                            continue
                        value = packet_on_wire_us(row, worker_rows)
                        if value is None:
                            continue
                        points.append((row["msg_size"], value))
                    points.sort()
                    if points:
                        series.append(
                            {
                                "name": f"{name}{_series_suffix(measure)}",
                                "x": [p[0] for p in points],
                                "y": [p[1] for p in points],
                            }
                        )
            if series:
                name = f"packet_on_wire_by_size_{safe_name(op)}_{level}.svg"
                (out_dir / name).write_text(
                    line_chart_svg(
                        f"Host packet-on-wire {STAT} latency by message size: {op} {level}",
                        f"packet on wire {STAT} (us)",
                        series,
                    )
                )
                generated.append(name)

            series = []
            for measure in measures:
                for stats in ["off", "on"]:
                    points = sorted(
                        [
                            (row["msg_size"], row["one_way_overhead_pct"])
                            for row in overhead
                            if row["op"] == op
                            and row["level"] == level
                            and row["proxy_stats"] == stats
                            and row.get("measure", "") == measure
                        ]
                    )
                    if points:
                        series.append(
                            {
                                "name": f"proxy stats {stats}{_series_suffix(measure)}",
                                "x": [p[0] for p in points],
                                "y": [p[1] for p in points],
                            }
                        )
            if series:
                name = f"stats_on_vs_off_overhead_{safe_name(op)}_{level}.svg"
                (out_dir / name).write_text(
                    line_chart_svg(
                        f"Proxy stats-on vs stats-off overhead by message size: {op} {level}",
                        "One-way overhead vs direct (%)",
                        series,
                        suffix="%",
                    )
                )
                generated.append(name)

    sizes = sorted({row["msg_size"] for row in rows})
    if selected_sizes is None:
        if len(sizes) <= 3:
            breakdown_sizes = sizes
        elif sizes:
            breakdown_sizes = sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
        else:
            breakdown_sizes = []
    else:
        available = set(sizes)
        breakdown_sizes = [size for size in selected_sizes if size in available]
    for size in breakdown_sizes:
        size_rows = [row for row in rows_for_size(rows, size) if row["role"] == "sender"]
        direct_rows = [row for row in size_rows if row["variant"] == "direct"]
        op_level_pairs = sorted(
            {(row["op"], row["level"]) for row in size_rows},
            key=lambda pair: (0 if pair[0] == "put" else 1, 0 if pair[1] == "thread" else 1),
        )

        for stats in ["off", "on"]:
            proxy_subset = [
                row for row in size_rows if row["variant"] == "proxy" and row["proxy_stats"] == stats
            ]
            if not proxy_subset:
                continue
            subset = direct_rows + proxy_subset
            name = f"direct_vs_proxy_stats_{stats}_{size}.svg"
            (out_dir / name).write_text(
                horizontal_path_svg(
                    f"Direct vs proxy GPU breakdown ({STAT}), stats {stats}, {size} bytes",
                    subset,
                )
            )
            generated.append(name)

            for op, level in op_level_pairs:
                ol_subset = [r for r in subset if r["op"] == op and r["level"] == level]
                if not ol_subset:
                    continue
                ol_name = f"direct_vs_proxy_stats_{stats}_{safe_name(op)}_{level}_{size}.svg"
                (out_dir / ol_name).write_text(
                    horizontal_path_svg(
                        f"Direct vs proxy GPU breakdown ({STAT}), {op} {level}, stats {stats}, {size} bytes",
                        ol_subset,
                    )
                )
                generated.append(ol_name)

        stats_on_subset = [
            row for row in size_rows if row["variant"] == "proxy" and row["proxy_stats"] == "on"
        ]
        if stats_on_subset:
            worker_subset = direct_rows + stats_on_subset
            name = f"direct_vs_proxy_worker_stats_on_{size}.svg"
            (out_dir / name).write_text(
                horizontal_path_svg(
                    f"Direct vs proxy GPU and worker breakdown ({STAT}), stats on, {size} bytes",
                    worker_subset,
                    worker_rows,
                )
            )
            generated.append(name)

            for op, level in op_level_pairs:
                ol_subset = [r for r in worker_subset if r["op"] == op and r["level"] == level]
                if not ol_subset:
                    continue
                ol_name = f"direct_vs_proxy_worker_stats_on_{safe_name(op)}_{level}_{size}.svg"
                (out_dir / ol_name).write_text(
                    horizontal_path_svg(
                        f"Direct vs proxy GPU and worker breakdown ({STAT}), {op} {level}, stats on, {size} bytes",
                        ol_subset,
                        worker_rows,
                    )
                )
                generated.append(ol_name)

            causal_name = f"causal_rtt_breakdown_stats_on_{size}.svg"
            causal_svg = four_lane_pingpong_svg(
                f"Causal RTT view ({STAT}), stats on, {size} bytes",
                rows,
                worker_rows,
                size,
                stats_filter="on",
            )
            if causal_svg:
                (out_dir / causal_name).write_text(causal_svg)
                generated.append(causal_name)

            for op, level in op_level_pairs:
                ol_rows = [r for r in rows if r["op"] == op and r["level"] == level]
                ol_causal_svg = four_lane_pingpong_svg(
                    f"Causal RTT view ({STAT}), {op} {level}, stats on, {size} bytes",
                    ol_rows,
                    worker_rows,
                    size,
                    stats_filter="on",
                )
                if not ol_causal_svg:
                    continue
                ol_causal_name = f"causal_rtt_breakdown_stats_on_{safe_name(op)}_{level}_{size}.svg"
                (out_dir / ol_causal_name).write_text(ol_causal_svg)
                generated.append(ol_causal_name)

        for stats in ["off", "on"]:
            hist_svg = phase_histogram_svg(
                f"Per-phase latency histograms ({STAT}), stats {stats}, {size} bytes",
                rows,
                worker_rows,
                size,
                stats_filter=stats,
            )
            if hist_svg:
                hist_name = f"phase_histograms_stats_{stats}_{size}.svg"
                (out_dir / hist_name).write_text(hist_svg)
                generated.append(hist_name)
                sidecar_rows = [
                    record
                    for record in build_phase_breakdown_rows(
                        [
                            row
                            for row in rows
                            if row["msg_size"] == size
                            and row["role"] == "sender"
                            and (row["variant"] == "direct" or row["proxy_stats"] == stats)
                        ],
                        [
                            w
                            for w in worker_rows
                            if w["msg_size"] == size
                            and w["role"] == "sender"
                            and w["proxy_stats"] == stats
                        ],
                    )
                    if record["hist_total"] > 0
                ]
                if sidecar_rows:
                    write_phase_breakdown_csv(
                        out_dir / f"phase_histograms_stats_{stats}_{size}.csv", sidecar_rows
                    )

            for op, level in op_level_pairs:
                ol_rows = [r for r in rows if r["op"] == op and r["level"] == level]
                ol_workers = [w for w in worker_rows if w["op"] == op and w["level"] == level]
                ol_hist_svg = phase_histogram_svg(
                    f"Per-phase latency histograms ({STAT}), {op} {level}, stats {stats}, {size} bytes",
                    ol_rows,
                    ol_workers,
                    size,
                    stats_filter=stats,
                )
                if not ol_hist_svg:
                    continue
                ol_hist_name = (
                    f"phase_histograms_stats_{stats}_{safe_name(op)}_{level}_{size}.svg"
                )
                (out_dir / ol_hist_name).write_text(ol_hist_svg)
                generated.append(ol_hist_name)
                ol_sidecar_rows = [
                    record
                    for record in build_phase_breakdown_rows(
                        [
                            row
                            for row in ol_rows
                            if row["msg_size"] == size
                            and row["role"] == "sender"
                            and (row["variant"] == "direct" or row["proxy_stats"] == stats)
                        ],
                        [
                            w
                            for w in ol_workers
                            if w["msg_size"] == size
                            and w["role"] == "sender"
                            and w["proxy_stats"] == stats
                        ],
                    )
                    if record["hist_total"] > 0
                ]
                if ol_sidecar_rows:
                    write_phase_breakdown_csv(
                        out_dir / f"phase_histograms_stats_{stats}_{safe_name(op)}_{level}_{size}.csv",
                        ol_sidecar_rows,
                    )

    return generated


def make_notebook(run_dir: Path, out_dir: Path, chart_names: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rel_run = run_dir.as_posix()
    sample_count = len(rows)
    sizes = sorted({row["msg_size"] for row in rows})
    ops = ", ".join(sorted({row["op"] for row in rows}))
    levels = ", ".join(sorted({row["level"] for row in rows}))
    chart_groups: dict[str, list[str]] = defaultdict(list)
    for name in chart_names:
        if name.startswith("phase_histograms"):
            chart_groups["Per-Phase Latency Histograms"].append(name)
        elif name.startswith("direct_vs_proxy_worker") or name.startswith("causal"):
            chart_groups["Direct vs Proxy with Worker Attribution"].append(name)
        elif name.startswith("direct_vs_proxy_stats_off"):
            chart_groups["Direct vs Proxy GPU Phases (stats off)"].append(name)
        elif name.startswith("direct_vs_proxy_stats_on"):
            chart_groups["Direct vs Proxy GPU Phases (stats on)"].append(name)
        elif (
            name.startswith("one_way_by_size")
            or name.startswith("stats_on_vs_off_overhead")
            or name.startswith("packet_on_wire_by_size")
        ):
            chart_groups["Cross-size Latency and Overhead"].append(name)
        else:
            chart_groups["Other Charts"].append(name)
    chart_md = "\n\n".join(
        [
            f"## {group}\n\n"
            + "\n\n".join(f"### {name}\n\n![{name}]({name})" for name in names)
            for group, names in chart_groups.items()
            if names
        ]
    )
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Active Pane Ping-pong Analysis\n",
                    "\n",
                    f"Run directory: `{rel_run}`\n\n",
                    f"Parsed `{sample_count}` sender/receiver run rows across sizes `{sizes}`.\n\n",
                    f"Operations: `{ops}`. Levels: `{levels}`.\n\n",
                    f"Breakdown charts use the **{STAT}** statistic per phase. Re-run with `--stat p50` for tail-robust medians or `--stat p99` for tail latency.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Timing Model\n",
                    "\n",
                    "New-format `[pingpong-stats]` rows decompose role-local GPU time as:\n\n",
                    "```text\n",
                    "sender RTT ~= gpu issue + gpu complete + gpu peer-wait + rtt unaccounted\n",
                    "sender one-way = sender RTT / 2\n",
                    "```\n\n",
                    "The conservative charts put each sender, receiver, and worker lane at local zero; they do not "
                    "claim synchronized clocks. The causal charts place the receiver return path inside the "
                    "sender-observed RTT as a heuristic alignment and show residual/gap segments where the data "
                    "does not explain the full interval.\n\n",
                    "Older table-style runs are parsed as legacy GPU micro-stages and are labeled with `legacy gpu` "
                    "prefixes so they are not confused with worker `prepare`/`submit` phases.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import csv\n",
                    f"analysis_dir = Path({str(out_dir)!r})\n",
                    "metrics = list(csv.DictReader((analysis_dir / 'metrics.csv').open()))\n",
                    "overhead = list(csv.DictReader((analysis_dir / 'proxy_overhead.csv').open()))\n",
                    "worker = list(csv.DictReader((analysis_dir / 'worker_stats.csv').open()))\n",
                    "runs = list(csv.DictReader((analysis_dir / 'runs.csv').open()))\n",
                    "phase_breakdown = list(csv.DictReader((analysis_dir / 'phase_breakdown.csv').open()))\n",
                    "packet_on_wire = list(csv.DictReader((analysis_dir / 'packet_on_wire.csv').open()))\n",
                    "{'metrics': len(metrics), 'overhead': len(overhead), 'worker': len(worker), 'runs': len(runs), 'phase_breakdown': len(phase_breakdown), 'packet_on_wire': len(packet_on_wire)}\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Sanity Tables\n",
                    "\n",
                    "The next cell computes quick checks from normalized CSVs: GPU phase sums versus RTT, "
                    "`one-way` versus `rtt / 2`, worker `post_submit` versus `post_progress + post_check + post_wait`, "
                    "and a heuristic sender `peer-wait` comparison against receiver return-operation sums.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from collections import defaultdict\n",
                    "metric = defaultdict(dict)\n",
                    "for row in metrics:\n",
                    "    key = tuple(row.get(k, '') for k in ['role','op','level','measure','proxy_stats','variant','msg_size'])\n",
                    "    metric[key][row['phase']] = float(row.get('avg_us') or 0.0)\n",
                    "gpu_checks = []\n",
                    "for key, phases in metric.items():\n",
                    "    rtt = phases.get('rtt')\n",
                    "    if rtt:\n",
                    "        phase_sum = phases.get('issue',0)+phases.get('complete',0)+phases.get('peer-wait',0)\n",
                    "        gpu_checks.append({'key': key, 'phase_sum_us': round(phase_sum,3), 'rtt_us': round(rtt,3), 'gap_us': round(rtt-phase_sum,3), 'one_way_gap_us': round(phases.get('one-way',0)-rtt/2,3)})\n",
                    "gpu_checks[:12]\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "worker_by_key = defaultdict(dict)\n",
                    "for row in worker:\n",
                    "    key = tuple(row.get(k, '') for k in ['role','op','level','measure','proxy_stats','variant','msg_size','worker'])\n",
                    "    worker_by_key[key][row['stage']] = float(row.get('avg_us') or 0.0)\n",
                    "worker_checks = []\n",
                    "for key, stages in worker_by_key.items():\n",
                    "    post_submit = stages.get('post_submit', 0.0)\n",
                    "    parts = stages.get('post_progress',0)+stages.get('post_check',0)+stages.get('post_wait',0)\n",
                    "    if post_submit or parts:\n",
                    "        worker_checks.append({'key': key, 'post_submit_us': round(post_submit,3), 'parts_us': round(parts,3), 'residual_us': round(post_submit-parts,3)})\n",
                    "worker_checks[:12]\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## CSV Outputs\n",
                    "\n",
                    "- [`metrics.csv`](metrics.csv): normalized role/phase metrics from `[pingpong-stats]` or legacy tables.\n",
                    "- [`worker_stats.csv`](worker_stats.csv): normalized CPU proxy worker stages, percentiles, histograms, and poll counters when stats are enabled.\n",
                    "- [`proxy_overhead.csv`](proxy_overhead.csv): sender RTT/one-way proxy overhead versus direct, plus `packet_on_wire` (gpu issue [+ worker dequeue + prepare + submit]) overhead.\n",
                    "- [`packet_on_wire.csv`](packet_on_wire.csv): per-row host-side send latency. Direct rows: `packet_on_wire_us == gpu_issue_us`. Proxy stats-on rows: `packet_on_wire_us == gpu_issue_us + worker_dequeue_us + worker_prepare_us + worker_submit_us`. Proxy stats-off rows leave the components blank because worker stats are only emitted when proxy stats are enabled.\n",
                    "- [`runs.csv`](runs.csv): joined sender/receiver file index.\n",
                    "- [`phase_breakdown.csv`](phase_breakdown.csv): unified long-format table backing every breakdown / histogram chart. One row per (config x phase x source) covering GPU phases (issue/complete/peer-wait + legacy + summary) and worker stages, with `n`, percentiles, and the nine histogram bucket counts. Each histogram SVG has a sibling `phase_histograms_stats_<off|on>_<size>.csv` filtered to that chart's rows.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Generated Charts\n\n", chart_md],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Sweep directory containing summary.csv and send/recv files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output analysis directory")
    parser.add_argument("--notebook-name", default="active_pane_analysis.ipynb")
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated detailed chart sizes. Defaults to smallest,middle,largest.",
    )
    parser.add_argument(
        "--stat",
        default="avg",
        choices=STAT_CHOICES,
        help="Which percentile/aggregate to use for chart bars and overlays (default: avg). Use p50 for a saner view when distributions have long tails.",
    )
    args = parser.parse_args()
    global STAT
    STAT = args.stat

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")
    out_dir = (args.out_dir or run_dir / "analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_summary(run_dir)
    script_run_rows = read_script_run_index(run_dir)
    metric_rows = parse_role_metrics(run_dir)
    normalized_metric_rows = flatten_metric_rows(metric_rows)
    worker_rows = parse_worker_stats(run_dir)
    overhead_rows = derive_overhead(metric_rows, worker_rows)
    packet_on_wire_rows = build_packet_on_wire_rows(metric_rows, worker_rows)
    runs_rows = derive_runs(run_dir, metric_rows, script_run_rows)
    selected_sizes = None
    if args.sizes:
        selected_sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    chart_names = generate_charts(out_dir, metric_rows, overhead_rows, worker_rows, selected_sizes)

    write_csv(out_dir / "summary_manifest.csv", summary_rows)
    write_csv(out_dir / "metrics.csv", normalized_metric_rows)
    write_csv(out_dir / "worker_stats.csv", worker_rows)
    write_csv(out_dir / "proxy_overhead.csv", overhead_rows)
    write_csv(out_dir / "packet_on_wire.csv", packet_on_wire_rows)
    write_csv(out_dir / "runs.csv", runs_rows)
    phase_breakdown_rows = build_phase_breakdown_rows(metric_rows, worker_rows)
    write_phase_breakdown_csv(out_dir / "phase_breakdown.csv", phase_breakdown_rows)
    (out_dir / "analysis_data.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "run_metrics": metric_rows,
                "metrics": normalized_metric_rows,
                "worker_stats": worker_rows,
                "proxy_overhead": overhead_rows,
                "packet_on_wire": packet_on_wire_rows,
                "runs": runs_rows,
                "charts": chart_names,
            },
            indent=2,
            sort_keys=True,
        )
    )
    notebook = make_notebook(run_dir, out_dir, chart_names, metric_rows)
    (out_dir / args.notebook_name).write_text(json.dumps(notebook, indent=2))

    print(f"analysis_dir={out_dir}")
    print(
        f"metrics_rows={len(normalized_metric_rows)} run_rows={len(runs_rows)} "
        f"worker_rows={len(worker_rows)} overhead_rows={len(overhead_rows)}"
    )
    print(f"charts={len(chart_names)} notebook={out_dir / args.notebook_name}")


if __name__ == "__main__":
    main()
