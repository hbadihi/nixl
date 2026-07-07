# Proxy EP elasticity validation (per-(rank,lane) ring isolation)

This validates the CPU device-proxy change that gives each `(destination rank, lane)` its own
proxy work ring, so a rank's failure/teardown can no longer poison rings shared with other ranks
(the elasticity guarantee for `connect_ranks` / `disconnect_ranks`). Revive of a reconnecting rank's
stale latch is **connection-driven**: it rides on the remote-memview (re)registration that EP already
performs on every connect/disconnect (`_nixl_ep_memory_views_create` → `prepMemView`). The runtime
diffs each element's remote agent per band and bumps a per-band generation when it changes; the owning
proxy worker reconciles lazily (`resetChannel`) on its next loop. No busy-wait, no manual reset call.

The **elastic LL suite** is what exercises this change end-to-end; HT only checks steady-state
ring-encoding correctness.

## 0. Prerequisites (cluster)

- **UCX v1.21.x built `--with-cuda`** (the proxy backend needs `ucp_device_remote_mem_list_create`,
  absent in 1.20.x), CUDA, and the NIXL deps. Full setup: see `BUILD_SETUP.md`.
- Env (paths are cluster-specific; export before building):
  ```bash
  export UCX_INSTALL_DIR=/path/to/ucx-install        # local, --with-cuda
  export CUDA_HOME=/usr/local/cuda
  export INSTALL_DIR=/path/to/nixl-deps
  export PKG_CONFIG_PATH=$UCX_INSTALL_DIR/lib/pkgconfig:$INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
  export LD_LIBRARY_PATH=$UCX_INSTALL_DIR/lib:$INSTALL_DIR/lib:$LD_LIBRARY_PATH
  ```
- A Python venv with `torch` + `numpy` (meson gates the EP example on `import torch`):
  ```bash
  source <repo>/.venv/bin/activate     # must expose torch to meson's python3
  ```

## 1. Build the proxy tree (and a ucx-direct baseline)

`proxy` is **never** chosen by `-Dgpu_device_api_backend=auto`; it must be explicit.
```bash
cd <repo>
# proxy
rm -rf build-proxy && meson setup build-proxy --prefix=$INSTALL_DIR \
  -Ducx_path=$UCX_INSTALL_DIR -Dcudapath_inc=$CUDA_HOME/include -Dcudapath_lib=$CUDA_HOME/lib64 \
  -Dgpu_device_api_backend=proxy -Dbuild_nixl_ep=true --buildtype=debug
ninja -C build-proxy && ninja -C build-proxy install
# ucx-direct baseline (regression check)
rm -rf build-ucx && meson setup build-ucx --prefix=$INSTALL_DIR \
  -Ducx_path=$UCX_INSTALL_DIR -Dcudapath_inc=$CUDA_HOME/include -Dcudapath_lib=$CUDA_HOME/lib64 \
  -Dgpu_device_api_backend=ucx -Dbuild_nixl_ep=true --buildtype=debug
ninja -C build-ucx && ninja -C build-ucx install
```
Verify the meson summary reads `UCX GPU Device API: YES`, `GPU-side compile: YES`,
`Host-side compile: YES`, and `Resolved backend: proxy` (resp. `ucx`).

## 2. Unit test (mechanism, no cluster needed)

```bash
build-proxy/test/gtest/unit/unit --gtest_filter='*Proxy*:*proxy*'
```
Expect all pass, including `MemviewReregisterRevivesLatchedChannel` (latch a band, re-register the
remote memview with that band's agent changed → the worker reconciles, latch clears, a later valid
submission completes, and an **unchanged** band is left latched), `MemviewRegisterNoopWhenEncodingDisabled`
(no revive when per-rank encoding is off), and `InitRejectsChannelCountNotMultipleOfChannelsPerRank`.

## 3. Elastic LL — expansion + contraction (the gate)

Single node with ≥8 GPUs. `expansion_contraction.json`
(`[[0,1,2,3], [0..7], [0,1,2,3,4,-6,7], [0..7]]`) exercises every path of this change:

| Phase | Action | Exercises |
|---|---|---|
| 0→1 | 4 → 8 | **expansion**: `connect_ranks` on new ranks → memview rebuild activates bands 4-7 |
| 2 | remove rank 5, **kill** rank 6 (`-6`) | **contraction + fault**: `disconnect_ranks(5)`; rank 6 self-kills → its rings may latch. Memview rebuild marks 5 & 6 absent (quiesce). Survivors (0-4,7) must still pass |
| 3 | re-add 5 & 6 | **revive**: memview rebuild flips bands 5 & 6 absent→present → generation bump → worker `resetChannel` clears rank 6's stale latch |

```bash
source <repo>/.venv/bin/activate
export PYTHONPATH=<repo>/build-proxy/examples/device/ep
cd <repo>/examples/device/ep
NIXL_LOG_LEVEL=DEBUG python3 tests/elastic/elastic.py \
  --plan tests/elastic/expansion_contraction.json --num-processes 8 \
  --evidence-output /tmp/ep_evidence_proxy
```
Run the simpler plans first to isolate the grow path: `single_expansion.json` (4→8),
`double_expansion.json` (4→6→8). For a pure clean-contraction case (no concurrent kill), add a plan
`shrink_then_grow.json = [[0,1,2,3,4,5,6,7],[0,1,2,3],[0,1,2,3,4,5,6,7]]`.

To sweep proxy worker configurations and get a pass/fail table in one shot, use the matrix runner
(it wraps `elastic.py`, defaults to `expansion_contraction.json`, and summarizes to
`$OUT_DIR/summary.tsv`):
```bash
export PYTHONPATH=<repo>/build-proxy/examples/device/ep
cd <repo>/examples/device/ep
bash tests/elastic/proxy_worker_matrix.sh
# override e.g.: UCX_WORKERS="1 2 4 8" PROXY_WORKERS="1" PLAN=tests/elastic/single_expansion.json \
#                NIXL_LOG_LEVEL=DEBUG bash tests/elastic/proxy_worker_matrix.sh
```
`NIXL_EP_PROXY_UCX_WORKERS` sets UCX host workers (QPs per peer; submission routes to
`channel_id % num_workers`) and `NIXL_EP_PROXY_WORKERS` sets the proxy drain-thread count; both
are independent of the fixed per-rank ring stride, so any combo is correctness-safe.

### Pass criteria
1. **Correctness per phase:** no NaN in `combined_x`; `diff < 1e-5` (bf16) / `9e-4` (fp8); mask-aware
   validation zeroes removed/killed ranks.
2. **Isolation:** phase 2 (rank 6 killed) still passes for the surviving ranks — a killed/removed
   rank does not stall or error the others. This is the core fix.
3. **Revive fired, targeted:** with `NIXL_LOG_LEVEL=DEBUG`, revive logs **only**
   `ProxyWorker::resetChannel: channel=… discarded=…` (band activation in the runtime is silent). Each
   band logs `discarded=0` as it first activates; on the phase-3 re-add, killed rank 6's band
   (channels `[6*cpr, 6*cpr+cpr)`) logs `discarded>0` (its stale latched entries). Bands of healthy
   ranks (0-4, 7) are **not** reset by topology changes that don't involve them.
4. **Proxy path real, not bypassed:** the `ep_proxy_evidence_v1` JSON records in
   `/tmp/ep_evidence_proxy` show `backend == "proxy"`, `proxy_context_published == true`,
   `proxy_activity_submitted_work_count > 0`, and an observed `ll_all_rdma_fallback_count`
   (LL proxy uses the all-RDMA fallback). A correctness-only pass without this is **inconclusive**.

## 4. HT correctness on proxy (secondary; needs 2 nodes)

HT asserts `num_ranks > 8`, so a single node is rejected. On 2 nodes (node0 IP `$IP0`), 8 ranks each:
```bash
# node0
WORLD_SIZE=2 RANK=0 MASTER_ADDR=$IP0 MASTER_PORT=8361 \
  PYTHONPATH=<repo>/build-proxy/examples/device/ep \
  python3 tests/test_ht.py --proxy-smoke --evidence-output /tmp/ht_ev
# node1
WORLD_SIZE=2 RANK=1 MASTER_ADDR=$IP0 MASTER_PORT=8361 \
  PYTHONPATH=<repo>/build-proxy/examples/device/ep \
  python3 tests/test_ht.py --proxy-smoke --tcp-server $IP0 --evidence-output /tmp/ht_ev
```
Pass: `calc_diff < 5e-6` (activations) / `< 1e-9` (top-k); evidence `validation_path="ht_proxy_smoke"`,
`backend="proxy"`, `proxy_activity>0`.

## 5. UCX-direct regression (confirm non-proxy unchanged)

Repeat §3/§4 from `build-ucx` (`PYTHONPATH=<repo>/build-ucx/examples/device/ep`). All proxy config is
`#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY`-gated, and the memview-driven band activation only runs inside
the proxy runtime's `prepMemView` (the UCX-direct path never calls it), so behavior must match
pre-change.
