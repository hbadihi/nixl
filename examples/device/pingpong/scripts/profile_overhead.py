#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# profile_overhead.py — compare the UCX-direct and CPU-proxy variants of the
# device pingpong benchmark.  Python port of profile_overhead.sh.
#
# Modes:
#   sweep          [--iters N] [--warmup N]                 msg-size sweep on both binaries
#   cluster-submit [--iters N] [--warmup N]                 SSH two-host submit-overhead sweep
#   slurm-submit   [--iters N] [--warmup N]                 Slurm/enroot submit-overhead sweep
#   nsys           [--size N]  [--iters N] [--warmup N]     capture an Nsight Systems trace
#   ucxinfo        [--size N]  [--iters N] [--warmup N]     dump UCX_PROTO_INFO for both
#   all                                               sweep + nsys + ucxinfo (defaults)
#
# Examples:
#   ./profile_overhead.py sweep
#   ./profile_overhead.py sweep --iters 5000 --warmup 500
#   ./profile_overhead.py nsys --size 8192 --iters 2000
#   ./profile_overhead.py ucxinfo --size 8 --iters 200
#   OUT_DIR=/tmp/run1 ./profile_overhead.py all
#
# Tunables (env vars, mirror the bash version):
#   REPO_ROOT   path to nixl repo root            (default: derived from script)
#   BUILD_DIR   meson build dir                   (default: $REPO_ROOT/build)
#   BIN_DIR     directory holding both binaries   (default: $BUILD_DIR/examples/device/pingpong)
#   RECV_GPU    GPU id for receiver               (default: 0)
#   SEND_GPU    GPU id for sender                 (default: 0; safe to share one GPU)
#   RECV_HOST   host receiver listens on          (default: 127.0.0.1)
#   BASE_PORT   first TCP port to use             (default: 19500)
#   SIZES       space-separated msg sizes         (default: 8 64 512 4096 32768 262144 1048576)
#   USE_WARP    if 1, pass --warp                 (default: 0)
#   OUT_DIR     output directory                  (default: $REPO_ROOT/profile_results/<ts>)
#   KILL_STALE  if 1, pkill stale bench procs     (default: 0)
#   RECV_WAIT_S receiver-cleanup timeout seconds  (default: 30)
#   NIXL_LOG_LEVEL    forwarded to bench          (default: FATAL)
#   NIXL_PROXY_STATS  forwarded to bench          (default: 1)
#
# Two-host submit sweep env vars:
#   SENDER_HOST     ssh target and advertised sender peer IP
#   RECEIVER_HOST   ssh target and advertised receiver peer IP
#   SENDER_GPU      sender GPU id                  (default: $SEND_GPU)
#   RECEIVER_GPU    receiver GPU id                (default: $RECV_GPU)
#   REMOTE_BIN_DIR  directory holding remote bins   (default: $BIN_DIR)
#   REMOTE_OUT_DIR  remote stdout/stderr directory  (default: $OUT_DIR)
#   SSH_CMD         ssh command prefix              (default: ssh)
#
# Slurm submit sweep env vars:
#   SENDER_NODE / RECEIVER_NODE        Slurm node names; default: first two allocated nodes
#   SENDER_IP / RECEIVER_IP            peer addresses passed to benchmark; default: node names
#   SENDER_CUDA_VISIBLE_DEVICES        sender CUDA_VISIBLE_DEVICES
#   RECEIVER_CUDA_VISIBLE_DEVICES      receiver CUDA_VISIBLE_DEVICES
#   SENDER_UCX_NET_DEVICES             sender UCX_NET_DEVICES
#   RECEIVER_UCX_NET_DEVICES           receiver UCX_NET_DEVICES
#   SLURM_CONTAINER_IMAGE              optional srun --container-image
#   SLURM_CONTAINER_MOUNTS             optional srun --container-mounts
#   SLURM_CONTAINER_WORKDIR            optional srun --container-workdir
#   SLURM_SRUN_ARGS                    extra args prepended to every srun
#
# Stdlib only — no pip dependencies.

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# ---------- defaults / paths -------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", SCRIPT_DIR.parents[3])).resolve()
BUILD_DIR = Path(os.environ.get("BUILD_DIR", REPO_ROOT / "build")).resolve()
BIN_DIR = Path(
    os.environ.get("BIN_DIR", BUILD_DIR / "examples" / "device" / "pingpong")
).resolve()

UCX_BIN = BIN_DIR / "nixl_device_pingpong"
PROXY_BIN = BIN_DIR / "nixl_device_pingpong_proxy"

RECV_GPU = os.environ.get("RECV_GPU", "0")
SEND_GPU = os.environ.get("SEND_GPU", "0")
RECV_HOST = os.environ.get("RECV_HOST", "127.0.0.1")
BASE_PORT = int(os.environ.get("BASE_PORT", "19500"))
USE_WARP = os.environ.get("USE_WARP", "0") == "1"
SIZES_STR = os.environ.get("SIZES", "8 64 512 4096 32768 262144 1048576")
SIZES = [int(s) for s in SIZES_STR.split()]

DEFAULT_ITERS = 2000
DEFAULT_WARMUP = 200

RECV_WAIT_S = int(os.environ.get("RECV_WAIT_S", "30"))
KILL_STALE = os.environ.get("KILL_STALE", "0") == "1"

# Default OUT_DIR is timestamped under $REPO_ROOT/profile_results.
_DEFAULT_OUT = (
    REPO_ROOT / "profile_results" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
)
OUT_DIR = Path(os.environ.get("OUT_DIR", _DEFAULT_OUT)).resolve()

SENDER_HOST = os.environ.get("SENDER_HOST", "")
RECEIVER_HOST = os.environ.get("RECEIVER_HOST", "")
SENDER_GPU = os.environ.get("SENDER_GPU", SEND_GPU)
RECEIVER_GPU = os.environ.get("RECEIVER_GPU", RECV_GPU)
REMOTE_BIN_DIR = os.environ.get("REMOTE_BIN_DIR", str(BIN_DIR))
REMOTE_OUT_DIR = os.environ.get("REMOTE_OUT_DIR", str(OUT_DIR))
SSH_CMD = os.environ.get("SSH_CMD", "ssh")
SRUN_CMD = os.environ.get("SRUN_CMD", "srun")
SLURM_CONTAINER_IMAGE = os.environ.get(
    "SLURM_CONTAINER_IMAGE", os.environ.get("CONTAINER_IMAGE", "")
)
SLURM_CONTAINER_MOUNTS = os.environ.get(
    "SLURM_CONTAINER_MOUNTS", os.environ.get("CONTAINER_MOUNTS", "")
)
SLURM_CONTAINER_WORKDIR = os.environ.get("SLURM_CONTAINER_WORKDIR", "")
SLURM_SRUN_ARGS = os.environ.get("SLURM_SRUN_ARGS", "")

# Inherited env we want to push down to children.  Quiets the bench logs by
# default — bench_host.cpp spins on prepMemView until remote metadata loads
# and would otherwise emit MB/s of ERROR lines on slow links.
_CHILD_ENV_DEFAULTS = {
    "NIXL_LOG_LEVEL": os.environ.get("NIXL_LOG_LEVEL", "FATAL"),
    "NIXL_PROXY_STATS": os.environ.get("NIXL_PROXY_STATS", "1"),
}


# ---------- small helpers ----------------------------------------------------


def log(msg: str) -> None:
    print(f"[{_dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def child_env(extra: Optional[dict] = None) -> dict:
    """Return os.environ overlaid with our defaults and any extras."""
    env = os.environ.copy()
    for k, v in _CHILD_ENV_DEFAULTS.items():
        env.setdefault(k, v)
    if extra:
        env.update(extra)
    return env


def check_binaries() -> None:
    missing = [
        b for b in (UCX_BIN, PROXY_BIN) if not (b.exists() and os.access(b, os.X_OK))
    ]
    if missing:
        for b in missing:
            print(f"ERROR: missing binary {b}", file=sys.stderr)
        print(
            f"Build with: ninja -C {BUILD_DIR} "
            "examples/device/pingpong/nixl_device_pingpong "
            "examples/device/pingpong/nixl_device_pingpong_proxy",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------- port allocation --------------------------------------------------


class _PortCursor:
    """Walk forward from BASE_PORT handing out free consecutive port pairs.

    Uses SO_REUSEADDR-bind probes rather than parsing `ss` output so the
    script doesn't depend on iproute2 being installed.
    """

    def __init__(self, base: int) -> None:
        self.cur = base
        self.base = base

    @staticmethod
    def _free(port: int) -> bool:
        # Try both IPv4 and IPv6 to mimic what the bench will actually do.
        for fam in (socket.AF_INET, socket.AF_INET6):
            try:
                with closing(socket.socket(fam, socket.SOCK_STREAM)) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("", port))
            except OSError:
                return False
        return True

    def next_pair(self) -> tuple[int, int]:
        while True:
            p_recv, p_send = self.cur, self.cur + 1
            self.cur += 2
            if self._free(p_recv) and self._free(p_send):
                return p_recv, p_send
            if self.cur > self.base + 200:
                raise RuntimeError(
                    f"no free port pair found in [{self.base}, {self.cur}]"
                )


_ports = _PortCursor(BASE_PORT)


# ---------- stale-process management ----------------------------------------

_STALE_PATTERN = r"nixl_device_pingpong(_proxy)?( |$)"


def _pgrep() -> list[int]:
    """Return PIDs of any running bench processes, [] if pgrep is unavailable."""
    if shutil.which("pgrep") is None:
        return []
    try:
        out = subprocess.run(
            ["pgrep", "-f", _STALE_PATTERN],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return []
    return [int(p) for p in out.split() if p.isdigit()]


def maybe_kill_stale() -> None:
    if not KILL_STALE:
        return
    victims = _pgrep()
    if victims:
        log(f"killing stale bench processes: {' '.join(map(str, victims))}")
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(1)


def warn_if_stale() -> None:
    stale = _pgrep()
    if stale:
        log(
            f"WARNING: existing bench processes are running (pids: {' '.join(map(str, stale))})"
        )
        log(
            "         re-run with KILL_STALE=1 to auto-kill, or:  pkill -9 -f nixl_device_pingpong"
        )


# ---------- one bench point --------------------------------------------------


@dataclass
class RunSpec:
    binary: Path
    size: int
    iters: int
    warmup: int
    tag: str
    nsys_rep: Optional[Path] = None  # if set, wrap sender in `nsys profile`
    extra_env: Optional[dict] = None  # extra env vars for both procs
    measure_submit: bool = False


def _build_args(
    role: str,
    listen_port: int,
    peer_port: int,
    spec: RunSpec,
    gpu: str,
    peer_ip: str = RECV_HOST,
    binary: Optional[Path] = None,
) -> list[str]:
    args = [
        str(binary or spec.binary),
        "--role",
        role,
        "--gpu",
        gpu,
        "--listen-port",
        str(listen_port),
        "--peer-ip",
        peer_ip,
        "--peer-port",
        str(peer_port),
        "--msg-size",
        str(spec.size),
        "--iters",
        str(spec.iters),
        "--warmup",
        str(spec.warmup),
    ]
    if USE_WARP:
        args.append("--warp")
    if spec.measure_submit:
        args.append("--measure-submit")
    return args


def run_one(spec: RunSpec) -> tuple[int, str]:
    """Run a single (binary, size) pair and return (rc, sender_stdout).

    Receiver runs in background, sender in foreground.  Stdout/stderr go to
    files under OUT_DIR named by `tag`.  Returns the sender exit code and
    its captured stdout (so callers can grep for RTT).
    """
    recv_out = OUT_DIR / f"{spec.tag}_recv.out"
    recv_err = OUT_DIR / f"{spec.tag}_recv.err"
    send_out = OUT_DIR / f"{spec.tag}_send.out"
    send_err = OUT_DIR / f"{spec.tag}_send.err"

    p_recv, p_send = _ports.next_pair()
    log(
        f"  run tag={spec.tag} size={spec.size} iters={spec.iters} "
        f"warmup={spec.warmup} ports=recv:{p_recv}/send:{p_send}"
    )

    env = child_env(spec.extra_env)

    recv_args = _build_args("receiver", p_recv, p_send, spec, RECV_GPU)
    send_args = _build_args("sender", p_send, p_recv, spec, SEND_GPU)

    # Receiver in background.
    with open(recv_out, "wb") as ro, open(recv_err, "wb") as re_:
        recv_proc = subprocess.Popen(recv_args, stdout=ro, stderr=re_, env=env)

    # Give the receiver a moment to bind & start listening before sender connects.
    time.sleep(1)

    # Sender in foreground (optionally wrapped in nsys).
    if spec.nsys_rep is not None:
        send_args = [
            "nsys",
            "profile",
            "-t",
            "cuda,nvtx,osrt",
            "-o",
            str(spec.nsys_rep),
            "--force-overwrite=true",
            *send_args,
        ]

    rc = 0
    try:
        with open(send_out, "wb") as so, open(send_err, "wb") as se:
            rc = subprocess.call(send_args, stdout=so, stderr=se, env=env)
    finally:
        # Reap receiver, but cap to RECV_WAIT_S so a hung peer can't deadlock us.
        deadline = time.monotonic() + RECV_WAIT_S
        while recv_proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if recv_proc.poll() is None:
            log(
                f"    receiver pid={recv_proc.pid} still alive after {RECV_WAIT_S}s — killing"
            )
            recv_proc.kill()
            try:
                recv_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    if rc != 0:
        log(
            f"    sender FAILED (rc={rc}) — see {send_out} {send_err} {recv_out} {recv_err}"
        )
        return rc, ""

    return 0, send_out.read_text(errors="replace")


# ---------- two-host submit sweep --------------------------------------------


def _ssh_prefix() -> list[str]:
    return shlex.split(SSH_CMD)


def _remote_path(binary: Path) -> Path:
    return Path(REMOTE_BIN_DIR) / binary.name


def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def _remote_cmd(host: str, argv: list[str], env: dict) -> list[str]:
    env_parts = [
        f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items())
    ]
    cmd = f"mkdir -p {shlex.quote(REMOTE_OUT_DIR)} && cd {shlex.quote(REMOTE_OUT_DIR)} && "
    if env_parts:
        cmd += "env " + " ".join(env_parts) + " "
    cmd += _shell_join(argv)
    return [*_ssh_prefix(), host, cmd]


def check_cluster_submit_config() -> None:
    missing = [
        name
        for name, value in (
            ("SENDER_HOST", SENDER_HOST),
            ("RECEIVER_HOST", RECEIVER_HOST),
            ("REMOTE_BIN_DIR", REMOTE_BIN_DIR),
        )
        if not value
    ]
    if missing:
        print("ERROR: cluster-submit requires " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    if not _ssh_prefix():
        print("ERROR: SSH_CMD must not be empty", file=sys.stderr)
        sys.exit(2)


def run_cluster_one(spec: RunSpec) -> tuple[int, str]:
    """Run one two-host point via ssh and return (sender rc, sender stdout)."""
    recv_out = OUT_DIR / f"{spec.tag}_recv.out"
    recv_err = OUT_DIR / f"{spec.tag}_recv.err"
    send_out = OUT_DIR / f"{spec.tag}_send.out"
    send_err = OUT_DIR / f"{spec.tag}_send.err"

    p_recv, p_send = _ports.next_pair()
    log(
        f"  cluster tag={spec.tag} size={spec.size} iters={spec.iters} "
        f"warmup={spec.warmup} ports=recv:{p_recv}/send:{p_send}"
    )

    env = child_env(spec.extra_env)
    remote_binary = _remote_path(spec.binary)
    recv_args = _build_args(
        "receiver",
        p_recv,
        p_send,
        spec,
        RECEIVER_GPU,
        peer_ip=SENDER_HOST,
        binary=remote_binary,
    )
    send_args = _build_args(
        "sender",
        p_send,
        p_recv,
        spec,
        SENDER_GPU,
        peer_ip=RECEIVER_HOST,
        binary=remote_binary,
    )

    recv_cmd = _remote_cmd(RECEIVER_HOST, recv_args, env)
    send_cmd = _remote_cmd(SENDER_HOST, send_args, env)

    with open(recv_out, "wb") as ro, open(recv_err, "wb") as re_:
        recv_proc = subprocess.Popen(recv_cmd, stdout=ro, stderr=re_)

    time.sleep(1)

    rc = 0
    try:
        with open(send_out, "wb") as so, open(send_err, "wb") as se:
            rc = subprocess.call(send_cmd, stdout=so, stderr=se)
    finally:
        deadline = time.monotonic() + RECV_WAIT_S
        while recv_proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if recv_proc.poll() is None:
            log(
                f"    remote receiver ssh pid={recv_proc.pid} still alive after "
                f"{RECV_WAIT_S}s - killing"
            )
            recv_proc.kill()
            try:
                recv_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    if rc != 0:
        log(
            f"    sender FAILED (rc={rc}) - see {send_out} {send_err} {recv_out} {recv_err}"
        )
        return rc, ""

    return 0, send_out.read_text(errors="replace")


# ---------- Slurm/enroot submit sweep ----------------------------------------


@dataclass
class SlurmConfig:
    sender_node: str
    receiver_node: str
    sender_ip: str
    receiver_ip: str
    sender_gpu: str
    receiver_gpu: str
    sender_cuda_visible_devices: str
    receiver_cuda_visible_devices: str
    sender_ucx_net_devices: str
    receiver_ucx_net_devices: str
    container_image: str
    container_mounts: str
    container_workdir: str
    extra_srun_args: list[str]
    capture_topology: bool = True


def _split_words(value: str) -> list[str]:
    return shlex.split(value) if value else []


def _slurm_allocated_nodes() -> list[str]:
    if os.environ.get("SLURM_JOB_NODELIST") and shutil.which("scontrol"):
        out = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        nodes = [line.strip() for line in out.splitlines() if line.strip()]
        if nodes:
            return nodes

    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    return [node]


def _slurm_container_args(cfg: SlurmConfig) -> list[str]:
    args: list[str] = []
    if cfg.container_image:
        args.append(f"--container-image={cfg.container_image}")
    if cfg.container_mounts:
        args.append(f"--container-mounts={cfg.container_mounts}")
    if cfg.container_workdir:
        args.append(f"--container-workdir={cfg.container_workdir}")
    return args


def _srun_prefix(node: str, cfg: SlurmConfig) -> list[str]:
    return [
        *_split_words(SRUN_CMD),
        "-N1",
        "-n1",
        "-w",
        node,
        *cfg.extra_srun_args,
        *_slurm_container_args(cfg),
    ]


def _role_env(cfg: SlurmConfig, role: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if role == "sender":
        cuda_visible = cfg.sender_cuda_visible_devices
        ucx_devices = cfg.sender_ucx_net_devices
    else:
        cuda_visible = cfg.receiver_cuda_visible_devices
        ucx_devices = cfg.receiver_ucx_net_devices

    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if ucx_devices:
        env["UCX_NET_DEVICES"] = ucx_devices
    return env


def _env_argv(env: dict[str, str], argv: list[str]) -> list[str]:
    if not env:
        return argv
    return ["env", *[f"{key}={value}" for key, value in sorted(env.items())], *argv]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_slurm_config(args: argparse.Namespace) -> SlurmConfig:
    nodes = _slurm_allocated_nodes()
    receiver_node = args.receiver_node or os.environ.get("RECEIVER_NODE", "")
    sender_node = args.sender_node or os.environ.get("SENDER_NODE", "")
    if not receiver_node and nodes:
        receiver_node = nodes[0]
    if not sender_node and len(nodes) > 1:
        sender_node = nodes[1]
    elif not sender_node and nodes:
        sender_node = nodes[0]

    if not sender_node or not receiver_node:
        print(
            "ERROR: slurm-submit needs two allocated nodes or explicit "
            "--sender-node/--receiver-node",
            file=sys.stderr,
        )
        sys.exit(2)

    sender_ip = args.sender_ip or os.environ.get("SENDER_IP", sender_node)
    receiver_ip = args.receiver_ip or os.environ.get("RECEIVER_IP", receiver_node)

    return SlurmConfig(
        sender_node=sender_node,
        receiver_node=receiver_node,
        sender_ip=sender_ip,
        receiver_ip=receiver_ip,
        sender_gpu=str(args.sender_gpu),
        receiver_gpu=str(args.receiver_gpu),
        sender_cuda_visible_devices=args.sender_cuda_visible_devices
        or os.environ.get("SENDER_CUDA_VISIBLE_DEVICES", ""),
        receiver_cuda_visible_devices=args.receiver_cuda_visible_devices
        or os.environ.get("RECEIVER_CUDA_VISIBLE_DEVICES", ""),
        sender_ucx_net_devices=args.sender_ucx_net_devices
        or os.environ.get("SENDER_UCX_NET_DEVICES", ""),
        receiver_ucx_net_devices=args.receiver_ucx_net_devices
        or os.environ.get("RECEIVER_UCX_NET_DEVICES", ""),
        container_image=args.container_image,
        container_mounts=args.container_mounts,
        container_workdir=args.container_workdir,
        extra_srun_args=[*args.srun_arg, *_split_words(SLURM_SRUN_ARGS)],
        capture_topology=not args.no_topo,
    )


def capture_slurm_topology(cfg: SlurmConfig) -> None:
    if not cfg.capture_topology:
        return

    for role, node in (("receiver", cfg.receiver_node), ("sender", cfg.sender_node)):
        out_path = OUT_DIR / f"topology_{role}_{_safe_name(node)}.out"
        err_path = OUT_DIR / f"topology_{role}_{_safe_name(node)}.err"
        env = _role_env(cfg, role)
        cmd = [
            *_srun_prefix(node, cfg),
            *_env_argv(env, ["bash", "-lc", "hostname; nvidia-smi topo -m"]),
        ]
        log(f"  capture {role} topology on {node} -> {out_path}")
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            rc = subprocess.call(cmd, stdout=out, stderr=err, env=child_env())
        if rc != 0:
            log(f"    topology capture failed for {role} rc={rc}; see {err_path}")


def run_slurm_one(spec: RunSpec, cfg: SlurmConfig) -> tuple[int, str]:
    recv_out = OUT_DIR / f"{spec.tag}_recv.out"
    recv_err = OUT_DIR / f"{spec.tag}_recv.err"
    send_out = OUT_DIR / f"{spec.tag}_send.out"
    send_err = OUT_DIR / f"{spec.tag}_send.err"

    p_recv, p_send = _ports.next_pair()
    log(
        f"  slurm tag={spec.tag} size={spec.size} iters={spec.iters} "
        f"warmup={spec.warmup} nodes=recv:{cfg.receiver_node}/send:{cfg.sender_node} "
        f"ports=recv:{p_recv}/send:{p_send}"
    )

    recv_args = _build_args(
        "receiver",
        p_recv,
        p_send,
        spec,
        cfg.receiver_gpu,
        peer_ip=cfg.sender_ip,
    )
    send_args = _build_args(
        "sender",
        p_send,
        p_recv,
        spec,
        cfg.sender_gpu,
        peer_ip=cfg.receiver_ip,
    )

    recv_cmd = [
        *_srun_prefix(cfg.receiver_node, cfg),
        *_env_argv(_role_env(cfg, "receiver"), recv_args),
    ]
    send_cmd = [
        *_srun_prefix(cfg.sender_node, cfg),
        *_env_argv(_role_env(cfg, "sender"), send_args),
    ]

    env = child_env(spec.extra_env)
    with open(recv_out, "wb") as ro, open(recv_err, "wb") as re_:
        recv_proc = subprocess.Popen(recv_cmd, stdout=ro, stderr=re_, env=env)

    time.sleep(1)

    rc = 0
    try:
        with open(send_out, "wb") as so, open(send_err, "wb") as se:
            rc = subprocess.call(send_cmd, stdout=so, stderr=se, env=env)
    finally:
        deadline = time.monotonic() + RECV_WAIT_S
        while recv_proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if recv_proc.poll() is None:
            log(
                f"    receiver srun pid={recv_proc.pid} still alive after {RECV_WAIT_S}s - killing"
            )
            recv_proc.kill()
            try:
                recv_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    if rc != 0:
        log(
            f"    sender FAILED (rc={rc}) - see {send_out} {send_err} {recv_out} {recv_err}"
        )
        return rc, ""

    return 0, send_out.read_text(errors="replace")


# ---------- output parsing ---------------------------------------------------

_FLOAT = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_KEYVAL_RE = re.compile(rf"\b(issue_us|submit_us|one_way_us|rtt_us)={_FLOAT}\b")
_RTT_RE = re.compile(rf"\bRTT={_FLOAT}\s*us\b")
_ONE_WAY_RE = re.compile(rf"\bone-way={_FLOAT}\s*us\b")
_TABLE_ROW_RE = re.compile(
    rf"^\s*(issue|submit|one-way|rtt)\s+{_FLOAT}(?:\s+|$)", re.MULTILINE
)


@dataclass
class BenchMetrics:
    issue_us: Optional[float] = None
    submit_us: Optional[float] = None
    one_way_us: Optional[float] = None
    rtt_us: Optional[float] = None


def _parse_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_metrics(text: str) -> BenchMetrics:
    """Parse benchmark stdout in table, key=value, and legacy RTT formats."""
    metrics = BenchMetrics()
    for key, value in _KEYVAL_RE.findall(text):
        parsed = _parse_float(value)
        if parsed is not None:
            setattr(metrics, key, parsed)

    for label, value in _TABLE_ROW_RE.findall(text):
        parsed = _parse_float(value)
        if parsed is None:
            continue
        if label == "issue" and metrics.issue_us is None:
            metrics.issue_us = parsed
        elif label == "submit" and metrics.submit_us is None:
            metrics.submit_us = parsed
        elif label == "one-way" and metrics.one_way_us is None:
            metrics.one_way_us = parsed
        elif label == "rtt" and metrics.rtt_us is None:
            metrics.rtt_us = parsed

    if metrics.one_way_us is None:
        m = _ONE_WAY_RE.search(text)
        if m:
            metrics.one_way_us = _parse_float(m.group(1))

    if metrics.rtt_us is None:
        m = _RTT_RE.search(text)
        if m:
            metrics.rtt_us = _parse_float(m.group(1))

    return metrics


def normalize_metrics(metrics: BenchMetrics, variant: str) -> BenchMetrics:
    if variant == "ucx" and metrics.submit_us is None:
        metrics.submit_us = metrics.issue_us
    return metrics


def fmt_metric(value: Optional[float]) -> str:
    return f"{value:.6f}" if value is not None else "NaN"


# ---------- mode: sweep ------------------------------------------------------


def do_sweep(iters: int, warmup: int) -> None:
    csv_path = OUT_DIR / "sweep.csv"
    log(f"sweep iters={iters} warmup={warmup} sizes=({' '.join(map(str, SIZES))})")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "msg_size",
                "iters",
                "warmup",
                "issue_us",
                "submit_us",
                "one_way_us",
                "rtt_us",
            ]
        )

        for size in SIZES:
            for variant, binary in (("ucx", UCX_BIN), ("proxy", PROXY_BIN)):
                tag = f"sweep_{variant}_{size}"
                spec = RunSpec(
                    binary=binary, size=size, iters=iters, warmup=warmup, tag=tag
                )
                rc, out = run_one(spec)
                if rc == 0:
                    metrics = normalize_metrics(parse_metrics(out), variant)
                    rtt_str = fmt_metric(metrics.rtt_us)
                    one_way_str = fmt_metric(metrics.one_way_us)
                    w.writerow(
                        [
                            variant,
                            size,
                            iters,
                            warmup,
                            fmt_metric(metrics.issue_us),
                            fmt_metric(metrics.submit_us),
                            one_way_str,
                            rtt_str,
                        ]
                    )
                    log(
                        f"    {variant} size={size} -> one-way={one_way_str} rtt={rtt_str} us"
                    )
                else:
                    w.writerow(
                        [variant, size, iters, warmup, "FAIL", "FAIL", "FAIL", "FAIL"]
                    )
                f.flush()
                time.sleep(1)

    log(f"wrote {csv_path}")
    print_sweep_summary(csv_path)


def print_sweep_summary(csv_path: Path) -> None:
    txt_path = OUT_DIR / "summary.txt"

    # variant -> size -> one-way latency (float) or None
    one_ways: dict[str, dict[int, Optional[float]]] = {"ucx": {}, "proxy": {}}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                size = int(row["msg_size"])
                val = _parse_float(row.get("one_way_us", ""))
            except (ValueError, KeyError):
                size = int(row.get("msg_size", "0") or 0)
                val = None
            one_ways.setdefault(row["variant"], {})[size] = val

    sizes = sorted(
        set(SIZES) | set(one_ways["ucx"].keys()) | set(one_ways["proxy"].keys())
    )

    lines: list[str] = [
        f"Sweep summary  (csv: {csv_path})",
        "---------------------------------------------------------------------------",
        f"  {'msg_size':>10}  {'ucx_oneway':>12}  {'proxy_oneway':>12}  {'delta_us':>12}  {'ratio':>10}",
    ]
    for s in sizes:
        u = one_ways["ucx"].get(s)
        p = one_ways["proxy"].get(s)
        if u and p and u > 0 and p > 0:
            lines.append(
                f"  {s:>10d}  {u:>12.2f}  {p:>12.2f}  {p - u:>12.2f}  {p / u:>9.2f}x"
            )
        else:
            u_s = f"{u:.2f}" if u else "FAIL"
            p_s = f"{p:.2f}" if p else "FAIL"
            lines.append(f"  {s:>10d}  {u_s:>12}  {p_s:>12}  {'-':>12}  {'-':>10}")

    body = "\n".join(lines) + "\n"
    sys.stdout.write(body)
    sys.stdout.flush()
    txt_path.write_text(body)
    log(f"wrote {txt_path}")


def do_cluster_submit(iters: int, warmup: int) -> None:
    check_cluster_submit_config()
    csv_path = OUT_DIR / "submit_sweep.csv"
    log(
        "cluster-submit "
        f"sender={SENDER_HOST} receiver={RECEIVER_HOST} "
        f"iters={iters} warmup={warmup} sizes=({' '.join(map(str, SIZES))})"
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "msg_size",
                "iters",
                "warmup",
                "issue_us",
                "submit_us",
                "one_way_us",
                "rtt_us",
            ]
        )

        for size in SIZES:
            for variant, binary in (("ucx", UCX_BIN), ("proxy", PROXY_BIN)):
                tag = f"submit_{variant}_{size}"
                spec = RunSpec(
                    binary=binary,
                    size=size,
                    iters=iters,
                    warmup=warmup,
                    tag=tag,
                    measure_submit=True,
                )
                rc, out = run_cluster_one(spec)
                if rc == 0:
                    metrics = normalize_metrics(parse_metrics(out), variant)
                    issue_str = fmt_metric(metrics.issue_us)
                    submit_str = fmt_metric(metrics.submit_us)
                    one_way_str = fmt_metric(metrics.one_way_us)
                    rtt_str = fmt_metric(metrics.rtt_us)
                    w.writerow(
                        [
                            variant,
                            size,
                            iters,
                            warmup,
                            issue_str,
                            submit_str,
                            one_way_str,
                            rtt_str,
                        ]
                    )
                    log(
                        f"    {variant} size={size} -> "
                        f"issue={issue_str} submit={submit_str} one-way={one_way_str} rtt={rtt_str} us"
                    )
                else:
                    w.writerow(
                        [variant, size, iters, warmup, "FAIL", "FAIL", "FAIL", "FAIL"]
                    )
                f.flush()
                time.sleep(1)

    log(f"wrote {csv_path}")
    print_submit_summary(csv_path)


def do_slurm_submit(args: argparse.Namespace) -> None:
    cfg = build_slurm_config(args)
    csv_path = OUT_DIR / "submit_sweep.csv"
    log(
        "slurm-submit "
        f"sender={cfg.sender_node}({cfg.sender_ip}) receiver={cfg.receiver_node}({cfg.receiver_ip}) "
        f"iters={args.iters} warmup={args.warmup} sizes=({' '.join(map(str, SIZES))})"
    )
    log(
        "  sender env: "
        f"CUDA_VISIBLE_DEVICES={cfg.sender_cuda_visible_devices or '<inherit>'} "
        f"UCX_NET_DEVICES={cfg.sender_ucx_net_devices or '<inherit>'}"
    )
    log(
        "  receiver env: "
        f"CUDA_VISIBLE_DEVICES={cfg.receiver_cuda_visible_devices or '<inherit>'} "
        f"UCX_NET_DEVICES={cfg.receiver_ucx_net_devices or '<inherit>'}"
    )

    capture_slurm_topology(cfg)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "msg_size",
                "iters",
                "warmup",
                "sender_node",
                "receiver_node",
                "sender_cuda_visible_devices",
                "receiver_cuda_visible_devices",
                "sender_ucx_net_devices",
                "receiver_ucx_net_devices",
                "issue_us",
                "submit_us",
                "one_way_us",
                "rtt_us",
            ]
        )

        for size in SIZES:
            for variant, binary in (("ucx", UCX_BIN), ("proxy", PROXY_BIN)):
                tag = f"slurm_submit_{variant}_{size}"
                spec = RunSpec(
                    binary=binary,
                    size=size,
                    iters=args.iters,
                    warmup=args.warmup,
                    tag=tag,
                    measure_submit=True,
                )
                rc, out = run_slurm_one(spec, cfg)
                base_row = [
                    variant,
                    size,
                    args.iters,
                    args.warmup,
                    cfg.sender_node,
                    cfg.receiver_node,
                    cfg.sender_cuda_visible_devices,
                    cfg.receiver_cuda_visible_devices,
                    cfg.sender_ucx_net_devices,
                    cfg.receiver_ucx_net_devices,
                ]
                if rc == 0:
                    metrics = normalize_metrics(parse_metrics(out), variant)
                    issue_str = fmt_metric(metrics.issue_us)
                    submit_str = fmt_metric(metrics.submit_us)
                    one_way_str = fmt_metric(metrics.one_way_us)
                    rtt_str = fmt_metric(metrics.rtt_us)
                    w.writerow([*base_row, issue_str, submit_str, one_way_str, rtt_str])
                    log(
                        f"    {variant} size={size} -> "
                        f"issue={issue_str} submit={submit_str} one-way={one_way_str} rtt={rtt_str} us"
                    )
                else:
                    w.writerow([*base_row, "FAIL", "FAIL", "FAIL", "FAIL"])
                f.flush()
                time.sleep(1)

    log(f"wrote {csv_path}")
    print_submit_summary(csv_path)


def _read_metric_by_variant(
    csv_path: Path, column: str
) -> dict[str, dict[int, Optional[float]]]:
    metrics: dict[str, dict[int, Optional[float]]] = {"ucx": {}, "proxy": {}}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                size = int(row["msg_size"])
            except (ValueError, KeyError):
                continue
            value = _parse_float(row.get(column, ""))
            metrics.setdefault(row.get("variant", ""), {})[size] = value
    return metrics


def print_submit_summary(csv_path: Path) -> None:
    txt_path = OUT_DIR / "submit_summary.txt"
    submits = _read_metric_by_variant(csv_path, "submit_us")
    sizes = sorted(
        set(SIZES) | set(submits["ucx"].keys()) | set(submits["proxy"].keys())
    )

    lines: list[str] = [
        f"Submit summary  (csv: {csv_path})",
        "---------------------------------------------------------------------------",
        f"  {'msg_size':>10}  {'ucx_submit':>12}  {'proxy_submit':>12}  {'delta_us':>12}  {'ratio':>10}",
    ]
    for size in sizes:
        ucx = submits["ucx"].get(size)
        proxy = submits["proxy"].get(size)
        if ucx and proxy and ucx > 0 and proxy > 0:
            lines.append(
                f"  {size:>10d}  {ucx:>12.2f}  {proxy:>12.2f}  "
                f"{proxy - ucx:>12.2f}  {proxy / ucx:>9.2f}x"
            )
        else:
            ucx_s = f"{ucx:.2f}" if ucx else "FAIL"
            proxy_s = f"{proxy:.2f}" if proxy else "FAIL"
            lines.append(
                f"  {size:>10d}  {ucx_s:>12}  {proxy_s:>12}  {'-':>12}  {'-':>10}"
            )

    body = "\n".join(lines) + "\n"
    sys.stdout.write(body)
    sys.stdout.flush()
    txt_path.write_text(body)
    log(f"wrote {txt_path}")


# ---------- mode: nsys -------------------------------------------------------


def do_nsys(size: int, iters: int, warmup: int) -> None:
    if shutil.which("nsys") is None:
        log("nsys not on PATH — skipping nsys mode")
        return
    log(f"nsys size={size} iters={iters} warmup={warmup}")
    for variant, binary in (("ucx", UCX_BIN), ("proxy", PROXY_BIN)):
        tag = f"nsys_{variant}_{size}"
        rep = OUT_DIR / tag  # nsys appends .nsys-rep
        spec = RunSpec(
            binary=binary, size=size, iters=iters, warmup=warmup, tag=tag, nsys_rep=rep
        )
        run_one(spec)
        log(f"  wrote {rep}.nsys-rep")
    log("open the .nsys-rep files in Nsight Systems to compare timelines")


# ---------- mode: ucxinfo ----------------------------------------------------


def do_ucxinfo(size: int, iters: int, warmup: int) -> None:
    log(f"ucxinfo size={size} iters={iters} warmup={warmup}")
    extra = {"UCX_LOG_LEVEL": "info", "UCX_PROTO_INFO": "y"}
    for variant, binary in (("ucx", UCX_BIN), ("proxy", PROXY_BIN)):
        tag = f"ucxinfo_{variant}"
        log(f"  {variant}: capturing UCX_PROTO_INFO")
        spec = RunSpec(
            binary=binary,
            size=size,
            iters=iters,
            warmup=warmup,
            tag=tag,
            extra_env=extra,
        )
        run_one(spec)
        log(f"    sender log: {OUT_DIR / (tag + '_send.err')}")
        log(f"    recv   log: {OUT_DIR / (tag + '_recv.err')}")


# ---------- entrypoint -------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode")

    sp = sub.add_parser("sweep", help="msg-size sweep on both binaries")
    sp.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    sp.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)

    cp = sub.add_parser("cluster-submit", help="two-host submit-overhead sweep")
    cp.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    cp.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)

    slp = sub.add_parser("slurm-submit", help="Slurm/enroot submit-overhead sweep")
    slp.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    slp.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    slp.add_argument("--sender-node", default=os.environ.get("SENDER_NODE", ""))
    slp.add_argument("--receiver-node", default=os.environ.get("RECEIVER_NODE", ""))
    slp.add_argument("--sender-ip", default=os.environ.get("SENDER_IP", ""))
    slp.add_argument("--receiver-ip", default=os.environ.get("RECEIVER_IP", ""))
    slp.add_argument("--sender-gpu", default=SENDER_GPU)
    slp.add_argument("--receiver-gpu", default=RECEIVER_GPU)
    slp.add_argument(
        "--sender-cuda-visible-devices",
        default=os.environ.get("SENDER_CUDA_VISIBLE_DEVICES", ""),
    )
    slp.add_argument(
        "--receiver-cuda-visible-devices",
        default=os.environ.get("RECEIVER_CUDA_VISIBLE_DEVICES", ""),
    )
    slp.add_argument(
        "--sender-ucx-net-devices",
        default=os.environ.get("SENDER_UCX_NET_DEVICES", ""),
    )
    slp.add_argument(
        "--receiver-ucx-net-devices",
        default=os.environ.get("RECEIVER_UCX_NET_DEVICES", ""),
    )
    slp.add_argument("--container-image", default=SLURM_CONTAINER_IMAGE)
    slp.add_argument("--container-mounts", default=SLURM_CONTAINER_MOUNTS)
    slp.add_argument("--container-workdir", default=SLURM_CONTAINER_WORKDIR)
    slp.add_argument(
        "--srun-arg",
        action="append",
        default=[],
        help="extra argument to pass to every srun; may be repeated",
    )
    slp.add_argument(
        "--no-topo",
        action="store_true",
        help="skip nvidia-smi topo -m capture on sender/receiver nodes",
    )

    np = sub.add_parser("nsys", help="capture an Nsight Systems trace")
    np.add_argument("--size", type=int, default=8192)
    np.add_argument("--iters", type=int, default=2000)
    np.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)

    up = sub.add_parser("ucxinfo", help="dump UCX_PROTO_INFO for both")
    up.add_argument("--size", type=int, default=8)
    up.add_argument("--iters", type=int, default=200)
    up.add_argument("--warmup", type=int, default=50)

    sub.add_parser("all", help="sweep + nsys + ucxinfo (defaults)")

    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    mode = args.mode or "sweep"

    if mode == "cluster-submit":
        check_cluster_submit_config()
    elif mode != "slurm-submit":
        check_binaries()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    maybe_kill_stale()
    warn_if_stale()

    if mode == "sweep":
        do_sweep(args.iters, args.warmup)
    elif mode == "cluster-submit":
        do_cluster_submit(args.iters, args.warmup)
    elif mode == "slurm-submit":
        do_slurm_submit(args)
    elif mode == "nsys":
        do_nsys(args.size, args.iters, args.warmup)
    elif mode == "ucxinfo":
        do_ucxinfo(args.size, args.iters, args.warmup)
    elif mode == "all":
        do_sweep(DEFAULT_ITERS, DEFAULT_WARMUP)
        do_nsys(8192, 2000, DEFAULT_WARMUP)
        do_ucxinfo(8, 200, 50)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        return 2

    log(f"results in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
