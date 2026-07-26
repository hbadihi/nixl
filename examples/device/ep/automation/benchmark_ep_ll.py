#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


EP_DIR = Path(__file__).resolve().parents[1]
AUTOMATION_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = AUTOMATION_DIR / "logs"
DEFAULT_PLAN = Path("tests/elastic/no_expansion.json")
DEFAULT_PYTHONPATH = "/workspace/external/nixl/install/lib/python3/dist-packages"
DEFAULT_GDP_PLUGIN_PATH = "/workspace/external/ucx-spcx-plugin/install/lib/ucx"
NUM_PROCESSES = 4
NUM_TOKENS = 128

BACKEND_TLS = {
    "rc_gda": "rc,rc_gda,tcp,self,sm,cuda_copy",
    "rc_gdp": "rc,rc_gdp,tcp,self,sm,cuda_copy",
}

DISPATCH_RE = re.compile(
    r"^\[rank (?P<rank>\d+)\] Dispatch bandwidth: "
    r"(?P<bandwidth>[0-9]+(?:\.[0-9]+)?) GB/s, "
    r"avg_t=(?P<avg_us>[0-9]+(?:\.[0-9]+)?) us, "
    r"min_t=(?P<min_us>[0-9]+(?:\.[0-9]+)?) us, "
    r"max_t=(?P<max_us>[0-9]+(?:\.[0-9]+)?) us$"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def channel_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 256 or parsed & (parsed - 1):
        raise argparse.ArgumentTypeError("must be a power of two in [1, 256]")
    return parsed


def timestamped_experiment_dir(log_root: Path, backend: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = log_root / f"{timestamp}_{backend}"
    experiment_dir = base
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = Path(f"{base}_{suffix}")
        suffix += 1
    experiment_dir.mkdir(parents=True)
    return experiment_dir


def latest_experiment_dir(log_root: Path) -> Path:
    if not log_root.exists():
        raise SystemExit(f"No log root found: {log_root}")
    experiments = sorted(path for path in log_root.iterdir() if path.is_dir())
    if not experiments:
        raise SystemExit(f"No experiment directories found under {log_root}")
    return experiments[-1]


def relpath(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def command_for_run(args: argparse.Namespace, experts: int) -> list[str]:
    return [
        args.python,
        "tests/elastic/elastic.py",
        "--plan",
        str(DEFAULT_PLAN),
        "--num-processes",
        str(NUM_PROCESSES),
        "--num-tokens",
        str(NUM_TOKENS),
        "--num-experts-per-rank",
        str(experts),
        "--dispatch-only",
    ]


def env_overrides_for_run(
    backend: str,
    channels: int,
    pythonpath: str | None,
    plugin_path: str,
) -> dict[str, str]:
    overrides = {
        "UCX_TLS": BACKEND_TLS[backend],
    }
    if pythonpath:
        overrides["PYTHONPATH"] = pythonpath
    if backend == "rc_gda":
        overrides["NIXL_EP_NUM_CHANNELS"] = str(channels)
    else:
        overrides.update(
            {
                "UCX_GDP_ENABLE": "y",
                "UCX_PLUGIN_PATH": plugin_path,
                "UCX_RC_GDP_NUM_CHANNELS": str(channels),
            }
        )
    return overrides


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_one(
    args: argparse.Namespace,
    experiment_dir: Path,
    experts: int,
    channels: int,
    repeat: int,
) -> dict[str, Any]:
    experts_dir = experiment_dir / f"ep{experts}"
    experts_dir.mkdir(exist_ok=True)
    log_path = experts_dir / f"channels{channels}_repeat{repeat}.log"
    status_path = experts_dir / f"channels{channels}_repeat{repeat}.json"
    command = command_for_run(args, experts)
    env_overrides = env_overrides_for_run(
        args.backend,
        channels,
        args.pythonpath,
        args.plugin_path,
    )
    env = os.environ.copy()
    env.update(env_overrides)

    run_info: dict[str, Any] = {
        "backend": args.backend,
        "experts_per_rank": experts,
        "channels": channels,
        "repeat": repeat,
        "command": command,
        "cwd": str(EP_DIR),
        "env": env_overrides,
        "log_file": relpath(log_path, experiment_dir),
        "status_file": relpath(status_path, experiment_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(
        f"Running backend={args.backend} ep={experts} "
        f"channels={channels} repeat={repeat}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# started_at: {run_info['started_at']}\n")
        log_file.write(f"# cwd: {EP_DIR}\n")
        log_file.write(f"# command: {' '.join(command)}\n")
        for key, value in sorted(env_overrides.items()):
            log_file.write(f"# env {key}={value}\n")
        log_file.write("\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=EP_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        printed_measurements = 0
        for line in process.stdout:
            log_file.write(line)
            match = DISPATCH_RE.match(line.rstrip("\n"))
            if match:
                printed_measurements += 1
                print(
                    f"  rank {match.group('rank')}: "
                    f"{match.group('bandwidth')} GB/s",
                    flush=True,
                )
        returncode = process.wait()

    run_info["returncode"] = returncode
    run_info["ended_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(status_path, run_info)
    if returncode != 0:
        print(
            f"Failed backend={args.backend} ep={experts} "
            f"channels={channels} repeat={repeat}: returncode={returncode}; "
            f"see {log_path}",
            flush=True,
        )
    elif printed_measurements == 0:
        print(
            f"No dispatch bandwidth lines found for backend={args.backend} "
            f"ep={experts} channels={channels} repeat={repeat}; see {log_path}",
            flush=True,
        )
    return run_info


def run_experiment(args: argparse.Namespace) -> Path:
    experiment_dir = timestamped_experiment_dir(args.log_root, args.backend)
    print(f"Experiment log directory: {experiment_dir}", flush=True)
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "experts": args.experts,
        "channels": args.channels,
        "repeats": args.repeats,
        "fixed_args": {
            "num_processes": NUM_PROCESSES,
            "num_tokens": NUM_TOKENS,
            "dispatch_only": True,
            "plan": str(DEFAULT_PLAN),
        },
        "python": args.python,
        "pythonpath": args.pythonpath,
        "plugin_path": args.plugin_path if args.backend == "rc_gdp" else None,
        "ep_dir": str(EP_DIR),
        "runs": [],
    }
    metadata_path = experiment_dir / "experiment.json"
    write_json(metadata_path, metadata)

    for experts in args.experts:
        for channels in args.channels:
            for repeat in range(args.repeats):
                run_info = run_one(args, experiment_dir, experts, channels, repeat)
                metadata["runs"].append(run_info)
                write_json(metadata_path, metadata)

    print(f"Experiment logs written to {experiment_dir}")
    return experiment_dir


def load_metadata(experiment_dir: Path) -> dict[str, Any]:
    metadata_path = experiment_dir / "experiment.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing experiment metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def parse_dispatch_log(log_path: Path) -> list[dict[str, Any]]:
    by_rank: dict[int, dict[str, Any]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = DISPATCH_RE.match(line)
        if not match:
            continue
        rank = int(match.group("rank"))
        if rank in by_rank:
            raise ValueError(f"duplicate dispatch measurement for rank {rank}")
        by_rank[rank] = {
            "rank": rank,
            "dispatch_bandwidth_gbps": float(match.group("bandwidth")),
            "avg_us": float(match.group("avg_us")),
            "min_us": float(match.group("min_us")),
            "max_us": float(match.group("max_us")),
        }

    expected_ranks = set(range(NUM_PROCESSES))
    actual_ranks = set(by_rank)
    if actual_ranks != expected_ranks:
        missing = sorted(expected_ranks - actual_ranks)
        extra = sorted(actual_ranks - expected_ranks)
        details = []
        if missing:
            details.append(f"missing ranks {missing}")
        if extra:
            details.append(f"unexpected ranks {extra}")
        raise ValueError(", ".join(details))

    return [by_rank[rank] for rank in sorted(by_rank)]


def write_measurements_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "experts_per_rank",
        "channels",
        "repeat",
        "rank",
        "returncode",
        "dispatch_bandwidth_gbps",
        "avg_us",
        "min_us",
        "max_us",
        "log_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "experts_per_rank",
        "channels",
        "completed_repeats",
        "mean_dispatch_bandwidth_gbps",
        "stdev_dispatch_bandwidth_gbps",
        "min_repeat_dispatch_bandwidth_gbps",
        "max_repeat_dispatch_bandwidth_gbps",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "experts_per_rank",
        "channels",
        "repeat",
        "returncode",
        "log_file",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_experiment(experiment_dir: Path) -> tuple[Path, Path]:
    metadata = load_metadata(experiment_dir)
    measurement_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    repeat_means: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    for run in metadata["runs"]:
        log_path = experiment_dir / run["log_file"]
        run_identity = {
            "backend": run["backend"],
            "experts_per_rank": run["experts_per_rank"],
            "channels": run["channels"],
            "repeat": run["repeat"],
            "returncode": run.get("returncode"),
            "log_file": run["log_file"],
        }
        if run.get("returncode") != 0:
            skipped_rows.append({**run_identity, "reason": "non-zero return code"})
            continue
        try:
            rank_measurements = parse_dispatch_log(log_path)
        except ValueError as exc:
            skipped_rows.append({**run_identity, "reason": str(exc)})
            continue

        rank_bandwidths = []
        for measurement in rank_measurements:
            rank_bandwidths.append(measurement["dispatch_bandwidth_gbps"])
            measurement_rows.append(
                {
                    **run_identity,
                    "rank": measurement["rank"],
                    "dispatch_bandwidth_gbps": measurement[
                        "dispatch_bandwidth_gbps"
                    ],
                    "avg_us": measurement["avg_us"],
                    "min_us": measurement["min_us"],
                    "max_us": measurement["max_us"],
                }
            )

        key = (run["backend"], run["experts_per_rank"], run["channels"])
        repeat_means[key].append(statistics.mean(rank_bandwidths))

    summary_rows: list[dict[str, Any]] = []
    for (backend, experts, channels), values in sorted(repeat_means.items()):
        summary_rows.append(
            {
                "backend": backend,
                "experts_per_rank": experts,
                "channels": channels,
                "completed_repeats": len(values),
                "mean_dispatch_bandwidth_gbps": statistics.mean(values),
                "stdev_dispatch_bandwidth_gbps": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "min_repeat_dispatch_bandwidth_gbps": min(values),
                "max_repeat_dispatch_bandwidth_gbps": max(values),
            }
        )

    measurements_path = experiment_dir / "measurements.csv"
    summary_path = experiment_dir / "summary.csv"
    skipped_path = experiment_dir / "skipped_runs.csv"
    write_measurements_csv(measurements_path, measurement_rows)
    write_summary_csv(summary_path, summary_rows)
    write_skipped_csv(skipped_path, skipped_rows)
    print(f"Wrote {measurements_path}")
    print(f"Wrote {summary_path}")
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} runs; details in {skipped_path}")
    return measurements_path, summary_path


def plot_summary(summary_csv: Path, output_png: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for plotting. Install matplotlib or run only "
            "the run/summarize subcommands."
        ) from exc

    with summary_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise SystemExit(f"No completed summary rows found in {summary_csv}")

    backends = sorted({row["backend"] for row in rows})
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["backend"], int(row["experts_per_rank"]))].append(row)

    channels = sorted({int(row["channels"]) for row in rows})
    channel_positions = {channel: index for index, channel in enumerate(channels)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for (backend, experts), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: int(row["channels"]))
        xs = [channel_positions[int(row["channels"])] for row in ordered]
        ys = [float(row["mean_dispatch_bandwidth_gbps"]) for row in ordered]
        yerr = [float(row["stdev_dispatch_bandwidth_gbps"]) for row in ordered]
        label = f"EP {experts}" if len(backends) == 1 else f"{backend} EP {experts}"
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=label)

    ax.set_xlabel("UCX device channels")
    ax.set_ylabel("Dispatch bandwidth (GB/s)")
    ax.set_title(f"NIXL EP LL dispatch bandwidth ({', '.join(backends)})")
    ax.set_xticks(list(channel_positions.values()))
    ax.set_xticklabels([str(channel) for channel in channels])
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Experts per rank")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_png}")
    return output_png


def resolve_experiment_dir(path: Path | None, log_root: Path) -> Path:
    return path if path is not None else latest_experiment_dir(log_root)


def cmd_run(args: argparse.Namespace) -> None:
    run_experiment(args)


def cmd_summarize(args: argparse.Namespace) -> None:
    experiment_dir = resolve_experiment_dir(args.experiment_dir, args.log_root)
    summarize_experiment(experiment_dir)


def cmd_plot(args: argparse.Namespace) -> None:
    experiment_dir = resolve_experiment_dir(args.experiment_dir, args.log_root)
    summary_csv = args.summary_csv or experiment_dir / "summary.csv"
    output_png = args.output_png or experiment_dir / "dispatch_bandwidth.png"
    plot_summary(summary_csv, output_png)


def cmd_all(args: argparse.Namespace) -> None:
    experiment_dir = run_experiment(args)
    _, summary_csv = summarize_experiment(experiment_dir)
    plot_summary(summary_csv, experiment_dir / "dispatch_bandwidth.png")


def add_log_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Root directory for timestamped experiment logs.",
    )


def add_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=sorted(BACKEND_TLS), required=True)
    parser.add_argument(
        "--experts",
        nargs="+",
        type=positive_int,
        required=True,
        help="One or more num-experts-per-rank values.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=channel_count,
        required=True,
        help="One or more power-of-two UCX channel counts in [1, 256].",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=1,
        help="Number of repeats per experts/channels combination.",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", "python3"),
        help="Python executable used to launch tests/elastic/elastic.py.",
    )
    parser.add_argument(
        "--pythonpath",
        default=os.environ.get("PYTHONPATH", DEFAULT_PYTHONPATH),
        help="PYTHONPATH for the benchmark process. Pass an empty string to inherit.",
    )
    parser.add_argument(
        "--plugin-path",
        default=os.environ.get("UCX_PLUGIN_PATH", DEFAULT_GDP_PLUGIN_PATH),
        help="UCX plugin path used by the rc_gdp backend.",
    )
    add_log_root_arg(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run, summarize, and plot NIXL EP LL dispatch experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the experiment matrix.")
    add_sweep_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Parse logs and write measurements.csv and summary.csv.",
    )
    summarize_parser.add_argument("experiment_dir", nargs="?", type=Path)
    add_log_root_arg(summarize_parser)
    summarize_parser.set_defaults(func=cmd_summarize)

    plot_parser = subparsers.add_parser("plot", help="Plot summary.csv to PNG.")
    plot_parser.add_argument("experiment_dir", nargs="?", type=Path)
    plot_parser.add_argument("--summary-csv", type=Path)
    plot_parser.add_argument("--output-png", type=Path)
    add_log_root_arg(plot_parser)
    plot_parser.set_defaults(func=cmd_plot)

    all_parser = subparsers.add_parser(
        "all",
        help="Run the matrix, summarize logs, and plot the PNG.",
    )
    add_sweep_args(all_parser)
    all_parser.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
