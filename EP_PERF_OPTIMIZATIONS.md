# NIXL EP over the LIBFABRIC CPU proxy: optimization log

Branch `efa-proxy/ep-perf-v2`, based on `efa-proxy/1t-perf-clean` (`8bac28bb`).
It replaces `efa-proxy/ep-perf-exp`, the first auto-research run, and keeps only
the changes that are allowed (see Rules). Drop this file before upstreaming.

## Rules

- Change only the libfabric CPU proxy (`src/plugins/libfabric`, `src/utils/libfabric`).
- The GPU producer (device proxy ring, `src/api/device/gpu`, `src/utils/device/proxy`)
  stays as it is: it is fast enough for the UCX proxy. The EP kernels stay as they are.
- No GPU-scope memory ordering where the protocol needs system scope.
- Keep the libfabric model: 1 rail per NIC, 1 proxy thread per rail.
- A change is adopted only if it is correct and it makes EP faster.

## Workload and method

NIXL EP, 8 ranks on one p5 node (2 processes x 4 ranks), 256 experts, 128 tokens,
hidden 7168, top-k 8, RDMA only, LIBFABRIC device proxy, 4 proxy threads, 1 rail
per NIC (RUNS.md section 2). Metric: dispatch + combine GB/s per rank, the mean of
8 ranks, EP correctness checks on. Runner: `.runlogs/ep30/ep.sh <tag>`. Two runs per
point. The spread between runs is about 1%, except where a range is given.

Leave-one-out: build the tip without one commit (`.runlogs/ep30v2/ablate.sh`)
and measure again. Each commit was first measured on top of the commits before it.
Because the gains compound, the leave-one-out drop shows if a commit is still
needed at the tip.

## Summary

Since RUNS.md exists, its evals are the source of truth: E2 (EP dispatch, 4 ranks) is the
main metric and E1 (nixlbench 1 -> 1) is the reference. The older rows below used the
8-rank dispatch + combine run.

| state | RUNS.md E2 dispatch GB/s | RUNS.md E1 GB/s | 8-rank d+c GB/s |
|---|---|---|---|
| `8bac28bb` baseline | 12.0 to 12.1 | 31.2 | 11.0 |
| this branch, tip | **17.9** (+48%) | **30.7** | about 19 |
| `efa-proxy/ep-perf-exp` tip (has the rejected device-side and EP commits) | - | - | 27.7 |

## Adopted

| commit | change | first gain (8-rank d+c, on the commits before it) | E2 leave-one-out at tip (17.9) |
|---|---|---|---|
| `091edd4e` | libfabric proxy: send each put to the peer NIC at the same rail position (was round-robin over the 4 NICs of the target GPU, 32:1 incast) | 11.0 -> 13.3 (+21%) | 15.8 (-12%) |
| `e01ec46d` | libfabric proxy: apply atomic adds on the target proxy thread (control message, batched streaming loads through GDRCopy; no provider-emulated atomic in `fi_cq_read`) | +13% (old stack) | 12.6 (-29%), together with the next commit, which needs it |
| `3ddb67d6` | libfabric proxy: send ready atomic adds to one target in one message (7.5 adds per message) | +14% (old stack) | 16.7 (-7%) |

Result of the leave-one-out runs: no adopted commit is redundant. On the old stack, the
only redundant commit was `rails_per_nic` (see below). On the 8-rank run the drops were
similar: -25%, -30%, -10%.

Other commits on the branch: `--dispatch-only` for the elastic test (RUNS.md E2) and the
kernel probes for RUNS.md metrics #3 to #5 (`NIXL_EP_PROBE=1`).

Open point: with `e01ec46d`, an atomic completes at the initiator when its message
is sent, before the target applies it. EP is correct with this, because it waits for
the counter value. A user that needs "applied" at quiesce needs an acknowledgement.

## Rejected: not correct or not allowed

| commit / idea | measured gain | reason |
|---|---|---|
| `31545fab` device proxy: publish a command after a GPU-scope fence | +14% (old stack) | Not correct: the NIC reads the payload through PCIe, so the GPU writes must be visible at system scope before the command is published. The PTX memory model does not promise that a GPU-scope fence does this. |
| `fef55491` device proxy: write a warp's command to the ring in one store (+ checksum) | +13% (old stack) | Not allowed: changes the GPU producer and the ring protocol shared with UCX. |
| `a29ff52e` nixl ep: send two dispatch tokens of one expert in one put | +6% (old stack); leave-one-out on the proxy-only stack: 21.7 -> 19.0 (-13%) | Not allowed: changes the EP kernel, not the proxy. |
| `7ccc104d` libfabric: open more than one rail per NIC (`rails_per_nic`) | +16% then, 0% after the target-applied atomics | Not allowed (1 rail per NIC). Also redundant: target-applied atomics removed the CPU limit that it worked around. |
| 2 or 4 rails per NIC with 4 or 8 threads, 4 rails per NIC for NIC load balance | 0% to -4% | Not allowed (1 rail per NIC, 1 thread per rail). |
| separate endpoint per NIC for atomic messages | not tested | Not allowed (one more rail per NIC). |
| provider atomic without the GPU read (libfabric hack) | +12% | Not correct in general: assumes the counter is 0. |
| atomic as an 8-byte RDMA write | same as target-applied atomics | Not correct in general: only valid for EP's counter use. |
| atomics rail-aligned (remote NIC = data NIC) | 0% | Only safe for EP. |
| no put fence before an atomic | +5% to +16% | Not correct: the signal can arrive before the data. |
| EP dispatch with 3 / 4 / all tokens per put, progressive pair puts, combine with 2 tokens per put, `__nanosleep` or GPU-scope loads in EP spin loops | see HANDOFF | Not allowed (EP kernel changes). |

The two device-side commits and the EP commit are worth about 45% together. This is
the gap between about 19 and 27.7 GB/s on the 8-rank run. The GPU publish cost under load is real: a
system-scope release takes about 158 us per command under NIC load, against 20 us at
GPU scope. By the rules, it is not a proxy problem.

## Rejected: no gain

Measured on `efa-proxy/ep-perf-exp` unless the row says "v2". The v2 rows were
measured on this branch while it still had the EP commit (tip 21.7). Flags and log tags are in
`.runlogs/ep30/HANDOFF.md` (old) and `.runlogs/ep30v2/exp_v2.patch` (v2). Re-test one
only if the bottleneck changes.

| idea | result |
|---|---|
| per-rail in-flight cap 16 / 32 / 64 / 128 | 10.2 / 13.2 / 16.1 / 16.5 vs 16.4 |
| v2: per-ring (= rail and peer) in-flight put cap 4 / 8 / 16 (`NIXL_EXP_RING_CAP`) | 14.9 / 19.2 / 21.4 vs 21.6 |
| v2: puts with `FI_TRANSMIT_COMPLETE` instead of `FI_DELIVERY_COMPLETE` (`NIXL_EXP_PUT_TC`) | 21.7 vs 21.6; device RDMA write gives the same completion |
| libfabric proxy: submit one command per ring in each pass (was 16) | E2 18.21 vs 17.42 GB/s (+4.5%), max_t 444 vs 530 us; E1 25.7 vs 30.7 GB/s (-18%): with 1 peer, a pass posts only 1 put per ring. Dropped for the E1 loss. An adaptive limit (more per ring when a thread has few active rings) is not tested. |
| v2: busy-wait 300 / 1000 / 3000 ns after a pass that dequeued nothing, less polling of the ring lines that the GPU writes (`NIXL_EXP_IDLE_NS`) | 21.4 / 19.5 / 15.5 vs 21.7 |
| peer rotation and per-peer interleave in submit | +1% |
| eager per-rail flush in submit (every 8 / 16 / 32 puts) | 0 to +2% |
| doorbell per post (no `FI_MORE`) | 0% |
| poll all owned rails in every pass | 0%, no change of the atomic latency tail |
| atomics always on the first rail of the thread | -20% |
| provider atomic with `FI_TRANSMIT_COMPLETE` | +5 to +8%, superseded by `e01ec46d` |
| target-side ordering: every put as `fi_writedata`, target counts arrivals | -15% (each writedata uses a posted receive) |
| same, with unsolicited write receive (chain test) | 0% |
| 8 or 16 channels with the same threads; 2 channels | 0% / -4% |
| one multi-SGE write for adjacent puts | not possible: efa-rdm splits iovs, efa-direct max message 8928 B |
| per-rank 12-core binding vs NUMA-node binding | 0% |
| dmabuf mapping off / default / pcie | 0% |
| commands per ring per pass 2 / 4 / 8 / 16 / 64 (8-rank d+c, old stack) | 27.9 / 27.7 / 27.6 / 27.4 / 27.0 vs 27.9 for 1 |

## Where the time goes (8-rank d+c run, 19.0 GB/s, with the per-ring submit commit)

CPU trace (`.runlogs/ep30v2/trace_v2.patch`, not on the branch; `tools/ph.py`, `tools/lat.py`,
`tools/inflight.py`), per phase, in us from the first command that the proxy dequeues
(mean of 80 phases):

| phase | puts | last put dequeued | last put done | last add applied | next phase starts |
|---|---|---|---|---|---|
| dispatch | 839 | 257 | 423 | 458 | 562 |
| combine | 785 | 229 | 368 | 406 | 504 |

- The GPU publishes the commands of a phase in 230 to 260 us. The NICs have 64 to 128
  puts in flight per rail for most of the phase, so the network, not the proxy, is
  the limit while the GPU publishes.
- Put latency (post to completion) is 110 us at the start of a phase and 200 us at the
  end. With 0 puts in flight on the rail it is still 45 us (p10) / 80 us (p50). The
  proxy reads completions without delay (p99 gap between trace events 8 us,
  completions come in bursts of about 5).
- The extra latency is incast: with 2 ranks (1 peer per NIC) the same measurement
  gives 28 / 41 us, near the 26 us of one idle EFA write.
- The same put latency shows with the device-side commits, so it does not come from
  the GPU publish path.
- The last atomic message takes about 35 us (fence done -> applied).
- GPU time per call (RUNS.md E2, kernel probes, rank 0): nixlPut p50 133 / p90 143 / p99 155 us,
  nixlAtomicAdd 38 / 44 / 55 us, no ring-full waits. The baseline proxy (`8bac28bb`) gives the
  same: nixlPut 133 / 148 / 162 us, nixlAtomicAdd 40 / 52 / 63 us. The proxy commits do not change
  the time that the GPU spends in the producer.
- From the last applied add to the next phase: about 100 us (GPU receive, next kernel,
  first system-scope publishes). With the device-side commits: about 22 us.

Ceilings (standalone `.runlogs/ep30/csrc/chain.c`, EP traffic with 1-token dispatch,
1 signal per peer, fenced, no NIXL):

| chain test | EP-equivalent GB/s |
|---|---|
| no GPU load | 27.7 |
| + 896 commands per phase written like the current producer (16-byte stores + system-scope release) | 25.5 |

## Experiments on this branch

| # | idea | result | verdict |
|---|---|---|---|
| 1 | per-ring in-flight put cap | worse at every cap | rejected, no gain |
| 2 | puts with `FI_TRANSMIT_COMPLETE` | 0% | rejected, no gain |
| 3 | backoff after idle passes | worse | rejected, no gain |
