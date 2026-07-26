# NIXL EP: Expert-Parallel Communication Example

## Overview

NIXL EP is a complete example implementation of expert-parallel communication for Mixture of Experts (MoE) models built on top of [NIXL](https://github.com/ai-dynamo/nixl)'s device API. It provides elastic scaling capabilities, enabling dynamic addition and removal of processes (ranks) during runtime without disrupting existing connections, and leverages NIXL's RDMA and NVLink support for optimal performance.

## Features
- **Dispatch and Combine support**: Supports dispatch and combine operations for MoE inference
- **RDMA and NVLink support**: Utilizes NIXL's abstractions to support both RDMA and NVLink transports for optimal performance
- **Elastic Scaling**: Dynamically add or remove ranks during runtime

## Buffer Initialization

NIXL EP provides a flexible buffer initialization pattern that supports dynamic rank management:

```python
import nixl_ep

# Initialize buffer with dynamic rank support
buffer = nixl_ep.Buffer(rank, explicitly_destroy=True)
buffer.update_memory_buffers(num_ranks, num_experts_per_rank, rdma_bytes)
buffer.connect_ranks(initial_ranks)

# Dispatch & Combine calls
buffer.dispatch(...)
buffer.combine(...)

# Later: Connect new ranks dynamically
buffer.connect_ranks(ranks)

# Dispatch & Combine calls
buffer.dispatch(...)
buffer.combine(...)

# Disconnect ranks when scaling down
buffer.disconnect_ranks(ranks)
```

## Key APIs

- `Buffer(rank_id, ...)`: Initialize the NIXL communication buffer
- `update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes, num_nvl_bytes=0)`: Prepare buffers for up to `num_ranks` ranks and `num_experts_per_rank` experts
- `connect_ranks(remote_ranks, activate=True)`: Establish NIXL connections to new peers (can be called multiple times); in low-latency mode, use `activate=False` to keep new peers masked until explicitly unmasked.
- `disconnect_ranks(remote_ranks)`: Clean up connections to departing peers

## Testing

The elastic test suite in `tests/elastic/` validates dynamic scaling capabilities:
- Plan files define scaling phases (representing an orchestrator)
- Tests validate correctness and measure bandwidth between scaling phases

**Example Plan** (`expansion_contraction.json`):
```json
[
  [0, 1, 2, 3],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 1, 2, 3, 4, 5]
]
```
This plan defines three phases:
- **Phase 0**: Initial state with ranks 0-3
- **Phase 1**: Ranks 4-7 are added dynamically (launched independently from initial ranks)
- **Phase 2**: Ranks 6-7 are removed dynamically

## Getting Started

#### Build NIXL with NIXL EP:

First, configure the pkg-config paths (only needed when dependencies are installed to non-default paths)

```bash
export PKG_CONFIG_PATH=<path to rdma-core install>/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<path to UCX install>/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<path to DOCA install>/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
```

Then, configure the NIXL plugin directory so it can find UCX plugin, and set the LD_LIBRARY_PATH so UCX can find rdma-core:
```bash
export NIXL_PLUGIN_DIR=<path to NIXL install directory>/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=<path to rdma-core install>/lib:$LD_LIBRARY_PATH
```

Build and install:

```bash
meson setup build \
    -Ducx_path=<path to UCX install> \
    -Dprefix=<path to NIXL install directory> \
    -Dbuildtype=release \
    -Dbuild_nixl_ep=true

cd build
ninja install
```


Finally, configure PYTHONPATH to use NIXL EP:
```bash
export PYTHONPATH=<path to NIXL build directory>/examples/device/ep
```

## EP LL Experiment Automation

The `automation/benchmark_ep_ll.py` CLI runs the low-latency dispatch benchmark
over a Cartesian product of experts-per-rank and UCX device channel counts. It
always uses `tests/elastic/no_expansion.json`, 4 worker processes, 128 tokens,
and `--dispatch-only`.

Run a complete sweep, summarize it, and generate a PNG:

```bash
python3 automation/benchmark_ep_ll.py all \
    --backend rc_gdp \
    --experts 32 64 \
    --channels 1 2 4 8 16 32 \
    --repeats 3
```

Each sweep creates a timestamped experiment directory under
`automation/logs/`, for example `automation/logs/20260726_144000_rc_gdp/`.
That directory contains:

- `experiment.json`: command, environment, fixed arguments, and per-run status.
- `ep<EP>/channels<N>_repeat<R>.log`: raw stdout/stderr for each matrix run.
- `ep<EP>/channels<N>_repeat<R>.json`: status metadata for each matrix run.
- `measurements.csv`: rank-level dispatch bandwidth measurements.
- `summary.csv`: repeat-level averages grouped by backend, EP, and channels.
- `skipped_runs.csv`: failed or incomplete runs that were omitted from summary.
- `dispatch_bandwidth.png`: channel count on the x-axis and dispatch bandwidth
  on the y-axis, with one series per EP value and error bars showing one sample
  standard deviation across repeats.

The three stages can also be run separately:

```bash
python3 automation/benchmark_ep_ll.py run \
    --backend rc_gda \
    --experts 64 \
    --channels 1 2 4 8 16 32 \
    --repeats 5

python3 automation/benchmark_ep_ll.py summarize automation/logs/<experiment>
python3 automation/benchmark_ep_ll.py plot automation/logs/<experiment>
```

Backend selection controls both `UCX_TLS` and the channel environment variable:

- `rc_gda` sets `UCX_TLS=rc,rc_gda,tcp,self,sm,cuda_copy` and
  `NIXL_EP_NUM_CHANNELS=<channels>`.
- `rc_gdp` sets `UCX_TLS=rc,rc_gdp,tcp,self,sm,cuda_copy`,
  `UCX_GDP_ENABLE=y`, `UCX_RC_GDP_NUM_CHANNELS=<channels>`, and
  `UCX_PLUGIN_PATH`. Override the plugin location with `--plugin-path`.

Channel counts must be exact powers of two in `[1, 256]`. The default
`PYTHONPATH` is inherited when set, otherwise it uses
`/workspace/external/nixl/install/lib/python3/dist-packages`; override it with
`--pythonpath`. Plotting requires Matplotlib.

Refer to [tests/elastic/README.md](tests/elastic/README.md) for detailed instructions on how to run the elastic test suite.
