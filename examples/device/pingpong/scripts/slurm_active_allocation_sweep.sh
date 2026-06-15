#!/usr/bin/env bash
set -euo pipefail

# Run from a shell inside the existing Slurm allocation.
# This creates new srun steps. It does not reuse already-attached container panes.
#
# Important UCX lesson from active-pane debugging:
# - UCX_NET_DEVICES must include both the NIC and the matching CUDA/NIC device
#   lane, e.g. mlx5_0:1,cuda0-mlx5_0:1.
# - Do not force UCX_TLS by default; UCX needs to choose rc_gda plus auxiliary
#   control lanes.

HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/workspace/external/nixl-cpu-proxy}"
SCRIPT="$HOST_REPO_ROOT/examples/device/pingpong/scripts/profile_overhead.py"
export BIN_DIR="${BIN_DIR:-$CONTAINER_REPO_ROOT/build/examples/device/pingpong}"
export BUILD_DIR="${BUILD_DIR:-$CONTAINER_REPO_ROOT/build}"

export OUT_DIR="${OUT_DIR:-$HOST_REPO_ROOT/profile_results/slurm-sweep-$(date +%Y%m%d-%H%M%S)}"
export SIZES="${SIZES:-8 64 512 4096 32768 262144 1048576}"

export RECEIVER_NODE="${RECEIVER_NODE:-pool0-01125}"
export SENDER_NODE="${SENDER_NODE:-pool0-01127}"
export RECEIVER_IP="${RECEIVER_IP:-100.126.3.41}"
export SENDER_IP="${SENDER_IP:-100.126.3.57}"

export RECEIVER_CUDA_VISIBLE_DEVICES="${RECEIVER_CUDA_VISIBLE_DEVICES:-0}"
export SENDER_CUDA_VISIBLE_DEVICES="${SENDER_CUDA_VISIBLE_DEVICES:-0}"

export RECEIVER_UCX_NET_DEVICES="${RECEIVER_UCX_NET_DEVICES:-mlx5_0:1,cuda0-mlx5_0:1}"
export SENDER_UCX_NET_DEVICES="${SENDER_UCX_NET_DEVICES:-mlx5_0:1,cuda0-mlx5_0:1}"
export UCX_MAX_RMA_RAILS="${UCX_MAX_RMA_RAILS:-1}"

export NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-FATAL}"
export NIXL_PROXY_STATS="${NIXL_PROXY_STATS:-1}"

# Keep UCX_TLS unset unless you are deliberately experimenting. Forcing a short
# list such as rc_gda,rc_mlx5,cuda_copy can break auxiliary transport selection.
unset UCX_TLS

DEFAULT_DRAFT_IMAGE="/lustre/fsw/portfolios/network/users/tdavidor/nixl_cpu_proxy_draft.sqsh"
DEFAULT_BASE_IMAGE="/lustre/fsw/portfolios/network/projects/network_research_advdev/users/tdavidor/enroot/images/nixl-cpu-proxy-ucx-atomic.sqsh"
if [[ -z "${SLURM_CONTAINER_IMAGE:-}" ]]; then
  if [[ -r "$DEFAULT_DRAFT_IMAGE" ]]; then
    export SLURM_CONTAINER_IMAGE="$DEFAULT_DRAFT_IMAGE"
  else
    export SLURM_CONTAINER_IMAGE="$DEFAULT_BASE_IMAGE"
  fi
fi
export SLURM_CONTAINER_MOUNTS="${SLURM_CONTAINER_MOUNTS:-/lustre/fsw/portfolios/network/users/tdavidor:/workspace/external}"
export SLURM_CONTAINER_WORKDIR="${SLURM_CONTAINER_WORKDIR:-$CONTAINER_REPO_ROOT}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_SRUN_ARGS="${SLURM_SRUN_ARGS:---jobid=$SLURM_JOB_ID --overlap --gpus=8}"
else
  echo "ERROR: SLURM_JOB_ID is empty. Run this from inside the allocation shell." >&2
  exit 2
fi

echo "slurm active allocation sweep"
echo "  host_repo=$HOST_REPO_ROOT"
echo "  container_repo=$CONTAINER_REPO_ROOT"
echo "  container_bin_dir=$BIN_DIR"
echo "  out_dir=$OUT_DIR"
echo "  container_image=$SLURM_CONTAINER_IMAGE"
echo "  receiver=$RECEIVER_NODE ip=$RECEIVER_IP ucx=$RECEIVER_UCX_NET_DEVICES"
echo "  sender=$SENDER_NODE ip=$SENDER_IP ucx=$SENDER_UCX_NET_DEVICES"
echo "  sizes=$SIZES"
echo "  UCX_TLS=<unset>"
echo "  SLURM_SRUN_ARGS=$SLURM_SRUN_ARGS"

python3 "$SCRIPT" slurm-submit --iters "${ITERS:-5000}" --warmup "${WARMUP:-500}"
