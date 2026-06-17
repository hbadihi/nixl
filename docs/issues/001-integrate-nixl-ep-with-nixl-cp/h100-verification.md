# H100 Verification Notes

This note records the supported-hardware validation used to close the local
L40S/sm_89 verification gap for Phase 1 of NIXL EP integration with the NIXL
CPU Proxy GPU device backend.

## Scope

Validated on H100/sm_90:

- UCX-direct EP baseline.
- Proxy backend import and backend selection.
- Proxy context publish.
- Single-node LL all-RDMA fallback through the CPU proxy.
- Single-node elastic LL expansion through the CPU proxy.
- Two-node HT proxy smoke through the CPU proxy.

Not covered by this correctness pass:

- Performance characterization.
- CPU proxy multi-worker infrastructure.
- Broader proxy channel scaling.
- Additional elastic failure/removal cases.
- Full production-size two-node benchmarking.

## Common Environment

Use these variables on every node before running the proxy build tests:

```bash
export KIND=proxy
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
export LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH
export CUDA_MODULE_LOADING=EAGER
export UCX_WARN_UNUSED_ENV_VARS=n
```

For UCX-direct baseline, use `KIND=ucx` and the UCX backend build.

## H100 Preflight

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

Expected hardware:

```text
NVIDIA H100 80GB HBM3
compute capability (9, 0)
```

## UCX-Direct Baseline

Build:

```bash
export KIND=ucx
meson setup build-$KIND \
  -Dbuild_nixl_ep=true \
  -Dgpu_device_api_backend=ucx \
  -Dnixl_cuda_arch_list=90 \
  -Ducx_path=$UCX_HOME \
  -Denable_plugins=UCX

ninja -C build-$KIND
```

Run:

```bash
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH \
python3 examples/device/ep/tests/ucx_direct_smoke.py \
  --evidence-output /tmp/nixl_ep_ucx_direct_h100.json
```

Expected evidence:

```text
backend: ucx
classification: accepted
correctness: pass
```

## Proxy Build And Import

Build:

```bash
export KIND=proxy
meson setup build-$KIND \
  -Dbuild_nixl_ep=true \
  -Dgpu_device_api_backend=proxy \
  -Dnixl_cuda_arch_list=90 \
  -Ducx_path=$UCX_HOME \
  -Denable_plugins=UCX

ninja -C build-$KIND
```

Import/backend check:

```bash
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH \
python3 -c "import nixl_ep, torch; print(nixl_ep.get_gpu_device_api_backend()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```text
proxy
NVIDIA H100 80GB HBM3
```

If a multi-node run fails with
`Buffer.update_memory_buffers() got an unexpected keyword argument 'proxy_lane_ceiling'`,
the node is importing a stale Python wrapper. Re-export `PYTHONPATH` on that
node and verify:

```bash
python3 - <<'PY'
import inspect
import nixl_ep
from nixl_ep import Buffer
from nixl_ep import nixl_ep_cpp

print("nixl_ep file:", nixl_ep.__file__)
print("cpp file:", nixl_ep_cpp.__file__)
print("backend:", nixl_ep.get_gpu_device_api_backend())
print("update_memory_buffers:", inspect.signature(Buffer.update_memory_buffers))
PY
```

The signature must include `proxy_lane_ceiling`.

## Proxy Context Publish Smoke

The first invalid one-rank smoke used `--num-tokens 1`, which fails the LL TMA
alignment guard. The second invalid one-rank smoke used one expert, which makes
the LL kernel launch with one SM and trips the `num_sms > 1` device assertion.

Use at least four tokens and two experts:

```bash
export KIND=proxy
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH \
CUDA_MODULE_LOADING=EAGER \
python3 examples/device/ep/tests/elastic/elastic.py \
  --plan examples/device/ep/tests/elastic/single_rank.json \
  --num-processes 1 \
  --num-tokens 4 \
  --hidden-dim 2048 \
  --num-experts-per-rank 2 \
  --num-topk 1 \
  --disable-ll-nvlink \
  --evidence-output /tmp/nixl_ep_proxy_publish_h100.json
```

Observed result:

```text
global_rank=0, local_rank=0 -> done
```

Observed evidence in
`/tmp/nixl_ep_proxy_publish_h100.rank0.phase0.json`:

```text
backend: proxy
correctness: pass
proxy_context_published: true
proxy_worker_count: 1
proxy_channel_count: 2
required_proxy_channels: 2
classification: inconclusive
reason: correctness passed but proxy worker activity was not observed
```

The `inconclusive` classification is expected for the single-rank publish smoke:
there is no remote rank, so there is no proxy worker data movement to observe.
The pass proves that proxy context publish works on H100.

## Single-Node 4-Rank LL Proxy Evidence

Run:

```bash
export KIND=proxy
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH \
CUDA_MODULE_LOADING=EAGER \
python3 examples/device/ep/tests/elastic/elastic.py \
  --plan examples/device/ep/tests/elastic/no_expansion.json \
  --num-processes 4 \
  --num-tokens 64 \
  --hidden-dim 2048 \
  --num-experts-per-rank 2 \
  --num-topk 2 \
  --disable-ll-nvlink \
  --timeout-ms 60000 \
  --evidence-output /tmp/nixl_ep_elastic_ll_h100.json
```

Observed result:

```text
ranks 0-3 completed phase 0 and exited cleanly
```

Evidence files:

```text
/tmp/nixl_ep_elastic_ll_h100.rank0.phase0.json
/tmp/nixl_ep_elastic_ll_h100.rank1.phase0.json
/tmp/nixl_ep_elastic_ll_h100.rank2.phase0.json
/tmp/nixl_ep_elastic_ll_h100.rank3.phase0.json
```

Observed evidence on all ranks:

```text
classification: accepted
backend: proxy
correctness: pass
validation_path: elastic_ll
proxy_context_published: true
proxy_activity_observed: true
ll_all_rdma_fallback_observed: true
proxy_worker_count: 1
proxy_channel_count: 2
required_proxy_channels: 2
```

## Single-Node 4-To-8 Elastic LL Expansion

Run:

```bash
export KIND=proxy
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH \
CUDA_MODULE_LOADING=EAGER \
python3 examples/device/ep/tests/elastic/elastic.py \
  --plan examples/device/ep/tests/elastic/single_expansion.json \
  --num-processes 8 \
  --num-tokens 64 \
  --hidden-dim 2048 \
  --num-experts-per-rank 2 \
  --num-topk 2 \
  --disable-ll-nvlink \
  --timeout-ms 60000 \
  --evidence-output /tmp/nixl_ep_elastic_ll_expansion_h100.json
```

Observed result:

```text
phase 0: ranks 0-3 completed
phase 1: ranks 0-7 completed after expansion
```

Evidence files:

```text
/tmp/nixl_ep_elastic_ll_expansion_h100.rank*.phase*.json
```

Observed evidence on all emitted rank/phase records:

```text
classification: accepted
backend: proxy
correctness: pass
validation_path: elastic_ll
proxy_context_published: true
proxy_activity_observed: true
ll_all_rdma_fallback_observed: true
proxy_worker_count: 1
proxy_channel_count: 2
required_proxy_channels: 2
```

## Two-Node 16-Rank HT Proxy Smoke

The existing HT proxy smoke requires more than eight total ranks and a rank
count divisible by eight. Use two H100 nodes with eight processes per node.

Set common variables on both nodes:

```bash
export KIND=proxy
export PYTHONPATH=$PWD/build-$KIND/examples/device/ep
export NIXL_PLUGIN_DIR=$PWD/build-$KIND/src/plugins/ucx
export LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH
export CUDA_MODULE_LOADING=EAGER
export UCX_WARN_UNUSED_ENV_VARS=n
export MASTER_ADDR=<node0_ip>
export MASTER_PORT=8361
export WORLD_SIZE=2
export NIXL_EP_PROXY_CHANNELS=12
```

Node 0:

```bash
export RANK=0
python3 examples/device/ep/tests/ht_proxy_smoke.py \
  --num-processes 8 \
  --num-tokens 128 \
  --hidden 2048 \
  --evidence-output /tmp/nixl_ep_ht_proxy_h100.json
```

Node 1:

```bash
export RANK=1
python3 examples/device/ep/tests/ht_proxy_smoke.py \
  --num-processes 8 \
  --num-tokens 128 \
  --hidden 2048 \
  --tcp-server $MASTER_ADDR \
  --evidence-output /tmp/nixl_ep_ht_proxy_h100.json
```

Evidence files:

```text
node 0: /tmp/nixl_ep_ht_proxy_h100.rank0.json through rank7.json
node 1: /tmp/nixl_ep_ht_proxy_h100.rank8.json through rank15.json
```

Observed evidence on all ranks 0-15:

```text
classification: accepted
backend: proxy
correctness: pass
validation_path: ht_proxy_smoke
num_nodes: 2
unsupported_single_node_fallback: false
proxy_context_published: true
proxy_activity_observed: true
proxy_worker_count: 1
proxy_channel_count: 12
required_proxy_channels: 12
```

## Final Correctness Status

H100 correctness validation passed for Phase 1:

```text
UCX-direct baseline: passed
Proxy backend import on H100: passed
Proxy context publish on H100: passed
Single-node 4-rank LL proxy: accepted
Single-node 4->8 elastic LL proxy: accepted
Two-node 16-rank HT proxy smoke: accepted
```

This is sufficient to accept the SDD verify gate for correctness-first Phase 1.
Performance, multi-worker proxy infrastructure, broader channel scaling, and
additional failure/elastic cases remain follow-up work.
