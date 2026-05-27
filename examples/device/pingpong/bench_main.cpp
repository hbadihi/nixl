// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bench_host.h"
#include "bench_kernel_iface.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ----------------------------------------------------------------------------
// Usage
// ----------------------------------------------------------------------------
static void
usage(const char *prog) {
    fprintf(stderr,
            "Two-process mode:\n"
            "  %s --role <sender|receiver>\n"
            "     --peer-ip     <ip>    peer's hostname or IP\n"
            "     --peer-port   <port>  peer's NIXL listen port\n"
            "     --listen-port <port>  our NIXL listen port\n"
            "    [--msg-size  <bytes>]  (default 8)\n"
            "    [--iters     <n>]      (default 1000)\n"
            "    [--warmup    <n>]      (default 100)\n"
            "    [--gpu       <id>]     (default 0)\n"
            "    [--warp]               use WARP level (default: THREAD)\n"
            "    [--op put|atomic-flag] (default put)\n"
            "    [--no-measure-submit]  skip GPU issue/submit timing metrics\n"
            "\n"
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
            "Single-process mode is not supported in the CPU-proxy build:\n"
            "the proxy device context is published into a process-wide\n"
            "__device__ pointer, so two agents in one process cannot coexist.\n",
            prog);
#else
            "Single-process mode (two threads, one GPU):\n"
            "  %s [--single-process]\n"
            "    [--msg-size  <bytes>]\n"
            "    [--iters     <n>]\n"
            "    [--warmup    <n>]\n"
            "    [--gpu       <id>]\n"
            "    [--warp]\n"
            "    [--op put|atomic-flag]\n"
            "    [--no-measure-submit]\n"
            "    [--base-port <port>]   loopback listen ports (default 12300);\n"
            "                           sender uses base, receiver uses base+1\n",
            prog, prog);
#endif
    exit(1);
}

static const char *
op_to_string(gpu_bench_op op) {
    switch (op) {
    case gpu_bench_op::Put:
        return "put";
    case gpu_bench_op::AtomicFlag:
        return "atomic-flag";
    }
    return "unknown";
}

static bool
parse_bench_op(const char *value, gpu_bench_op &op) {
    if (!strcmp(value, "put")) {
        op = gpu_bench_op::Put;
        return true;
    }
    if (!strcmp(value, "atomic-flag")) {
        op = gpu_bench_op::AtomicFlag;
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// Per-phase GPU timing stats: allocation, sample reduction, and worker-style
// output. The output format is intentionally identical to the host-side worker
// ([src/core/device_proxy/proxy_worker.cpp] printStats()) so the two halves of
// a run can be read side-by-side and grepped uniformly.
//
// Each PhaseStats owns one device-resident gpu_cycle_stats summary struct plus
// one device-side raw-sample buffer (sized to num_iters). After the kernel
// runs, samples are copied to host and used to compute nearest-rank
// percentiles (p50/p90/p99) and a microsecond histogram with the same bucket
// bounds as the worker.
// ----------------------------------------------------------------------------

// Keep in sync with kHistUpperBoundNs / kHistLabels in
// src/core/device_proxy/proxy_worker.cpp (bucket bounds expressed in µs here
// since GPU samples are converted from cycles to µs before bucketizing).
static constexpr std::array<double, 8> kHistUpperBoundUs = {
    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
static constexpr std::array<const char *, 9> kHistLabels = {
    "<0.1", "<0.5", "<1", "<2", "<5", "<10", "<50", "<100", ">=100"};

struct PhaseStats {
    const char      *name      = nullptr;
    gpu_cycle_stats *d_stats   = nullptr; // device summary struct
    uint64_t        *d_samples = nullptr; // device buffer of raw cycle samples
    uint64_t         capacity  = 0;
};

static bool
alloc_phase_stats(PhaseStats &p, const char *name, uint64_t capacity, const char *role) {
    p.name     = name;
    p.capacity = capacity;
    if (cudaMalloc(&p.d_stats, sizeof(gpu_cycle_stats)) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc %s summary failed\n", role, name);
        return false;
    }
    if (cudaMalloc(&p.d_samples, capacity * sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc %s samples (%llu entries) failed\n",
                role, name, static_cast<unsigned long long>(capacity));
        cudaFree(p.d_stats);
        p.d_stats = nullptr;
        return false;
    }
    cudaMemset(p.d_stats,   0, sizeof(gpu_cycle_stats));
    cudaMemset(p.d_samples, 0, capacity * sizeof(uint64_t));

    gpu_cycle_stats h_init{};
    h_init.samples  = p.d_samples;
    h_init.capacity = capacity;
    cudaMemcpy(p.d_stats, &h_init, sizeof(gpu_cycle_stats), cudaMemcpyHostToDevice);
    return true;
}

static void
free_phase_stats(PhaseStats &p) {
    cudaFree(p.d_stats);
    cudaFree(p.d_samples);
    p.d_stats   = nullptr;
    p.d_samples = nullptr;
}

static double
percentile_nearest_rank_us(const std::vector<double> &sorted_samples_us, double p) {
    if (sorted_samples_us.empty()) {
        return 0.0;
    }
    const size_t n    = sorted_samples_us.size();
    const size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
    const size_t idx  = rank == 0 ? 0 : rank - 1;
    return sorted_samples_us[std::min(idx, n - 1)];
}

static size_t
histogram_bucket_index_us(double us) {
    for (size_t i = 0; i < kHistUpperBoundUs.size(); ++i) {
        if (us < kHistUpperBoundUs[i]) {
            return i;
        }
    }
    return kHistLabels.size() - 1;
}

struct PhaseSummary {
    const char *name = nullptr;
    uint64_t count = 0;
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p90_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double stddev_us = 0.0;
    std::array<uint64_t, 9> hist{};
};

static PhaseSummary
summarize_phase(const PhaseStats &p, double clock_hz, double scale = 1.0) {
    PhaseSummary out;
    out.name = p.name;
    if (p.d_stats == nullptr) {
        return out;
    }

    gpu_cycle_stats h_stats{};
    cudaMemcpy(&h_stats, p.d_stats, sizeof(gpu_cycle_stats), cudaMemcpyDeviceToHost);
    out.count = h_stats.count;
    if (h_stats.count == 0) {
        return out;
    }

    const uint64_t n_kept = std::min<uint64_t>(h_stats.count, p.capacity);
    std::vector<uint64_t> cycles(n_kept);
    if (n_kept > 0 && p.d_samples != nullptr) {
        cudaMemcpy(cycles.data(), p.d_samples,
                   n_kept * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    const double cycles_to_us = 1e6 / clock_hz * scale;
    std::vector<double> samples_us;
    samples_us.reserve(n_kept);
    for (uint64_t c : cycles) {
        samples_us.push_back(static_cast<double>(c) * cycles_to_us);
    }
    std::sort(samples_us.begin(), samples_us.end());

    const double count_d    = static_cast<double>(h_stats.count);
    const double avg_cycles = static_cast<double>(h_stats.sum) / count_d;
    out.avg_us = avg_cycles * cycles_to_us;
    out.min_us = static_cast<double>(h_stats.min) * cycles_to_us;
    out.max_us = static_cast<double>(h_stats.max) * cycles_to_us;

    // Sample stddev with (n-1) divisor, matching the worker's sampleStddevUs.
    // M2 = sum_sq - n * mean^2; sample variance = M2 / (n - 1).
    if (h_stats.count > 1) {
        const double m2_cycles =
            std::fmax(0.0, h_stats.sum_sq - count_d * avg_cycles * avg_cycles);
        const double sample_variance_cycles = m2_cycles / (count_d - 1.0);
        out.stddev_us = std::sqrt(sample_variance_cycles) * cycles_to_us;
    }

    out.p50_us = percentile_nearest_rank_us(samples_us, 50.0);
    out.p90_us = percentile_nearest_rank_us(samples_us, 90.0);
    out.p99_us = percentile_nearest_rank_us(samples_us, 99.0);

    for (double us : samples_us) {
        ++out.hist[histogram_bucket_index_us(us)];
    }

    return out;
}

// Emit two lines per phase in the same format as
// [proxy-worker-stats][wN] <name> ... from the worker's printStats().
static void
print_phase_summary_lines(const PhaseSummary &s) {
    if (s.name == nullptr) {
        return;
    }
    if (s.count == 0) {
        printf("[pingpong-stats] %-11s n=0\n", s.name);
        return;
    }

    printf("[pingpong-stats] %-11s n=%llu avg=%9.3f us p50=%9.3f us "
           "p90=%9.3f us p99=%9.3f us min=%9.3f us max=%9.3f us stddev=%9.3f us\n",
           s.name,
           static_cast<unsigned long long>(s.count),
           s.avg_us, s.p50_us, s.p90_us, s.p99_us,
           s.min_us, s.max_us, s.stddev_us);

    char hist_line[256];
    int offset = 0;
    for (size_t i = 0; i < s.hist.size(); ++i) {
        offset += std::snprintf(hist_line + offset,
                                sizeof(hist_line) - static_cast<size_t>(offset),
                                "%s%s:%llu",
                                i == 0 ? "" : " ",
                                kHistLabels[i],
                                static_cast<unsigned long long>(s.hist[i]));
    }
    printf("[pingpong-stats] %-11s hist_us=%s\n", s.name, hist_line);
}

// Print an ASCII bar chart showing the average per-phase contribution to the
// sender's RTT. Bar width is scaled so the longest phase fills the bar; the
// `share` column shows each phase as a fraction of the sum of all phases in
// the chart, and a trailing line cross-checks the sum against the measured
// rtt avg so the reader can see how much of the RTT was unaccounted for.
//
// The row set is identical for the proxy and UCX device builds:
//   issue + complete + peer-wait ≈ rtt
// The proxy worker's internal stage acknowledgements
// (dequeued/prepared/submitted) are intentionally omitted from the GPU-side
// output: polling them from the GPU would inject a PCIe round-trip into the
// timed loop. The internal worker phases remain visible in the worker's own
// [proxy-worker-stats] block.
static void
print_breakdown_chart(const std::vector<PhaseSummary> &phases,
                      const PhaseSummary &rtt) {
    constexpr int kBarWidth = 40;

    double sum_avg = 0.0;
    double max_avg = 0.0;
    for (const PhaseSummary &p : phases) {
        if (p.count == 0) continue;
        sum_avg += p.avg_us;
        max_avg = std::fmax(max_avg, p.avg_us);
    }
    if (max_avg <= 0.0) {
        return;
    }

    printf("[pingpong-stats] breakdown (avg us, bar width = %d chars at the longest phase)\n",
           kBarWidth);

    for (const PhaseSummary &p : phases) {
        if (p.count == 0) continue;
        const double frac_bar  = p.avg_us / max_avg;
        const double frac_sum  = sum_avg > 0.0 ? p.avg_us / sum_avg : 0.0;
        const int    bar_chars =
            std::max(0, std::min(kBarWidth, static_cast<int>(std::lround(frac_bar * kBarWidth))));

        char bar[kBarWidth + 1];
        for (int i = 0; i < kBarWidth; ++i) {
            bar[i] = (i < bar_chars) ? '#' : '.';
        }
        bar[kBarWidth] = '\0';

        printf("[pingpong-stats]   %-11s |%s| %9.3f us  (%5.1f%%)\n",
               p.name, bar, p.avg_us, frac_sum * 100.0);
    }

    const double unaccounted_us  = rtt.avg_us - sum_avg;
    const double unaccounted_pct = rtt.avg_us > 0.0 ? unaccounted_us / rtt.avg_us * 100.0 : 0.0;
    printf("[pingpong-stats]   sum_phases  %9.3f us\n", sum_avg);
    printf("[pingpong-stats]   rtt_avg     %9.3f us  (unaccounted %+9.3f us; %+5.1f%%)\n",
           rtt.avg_us, unaccounted_us, unaccounted_pct);
}

// Holds every per-phase PhaseStats owned by a sender. Phases populated only
// when the corresponding measurement is enabled by the caller; unused phases
// stay default-constructed (d_stats == nullptr) and are silently skipped by
// print_phase_stats().
struct SenderPhaseStats {
    PhaseStats issue;
    PhaseStats completion;
    PhaseStats peer_wait;
    PhaseStats rtt;
};

static void
free_sender_phase_stats(SenderPhaseStats &s) {
    free_phase_stats(s.issue);
    free_phase_stats(s.completion);
    free_phase_stats(s.peer_wait);
    free_phase_stats(s.rtt);
}

// Fan SenderPhaseStats out into gpu_bench_ctx (the kernel expects raw
// device pointers, one per phase).
static void
attach_phase_stats(gpu_bench_ctx &kctx, const SenderPhaseStats &s) {
    kctx.issue_stats      = s.issue.d_stats;
    kctx.completion_stats = s.completion.d_stats;
    kctx.peer_wait_stats  = s.peer_wait.d_stats;
    kctx.rtt_stats        = s.rtt.d_stats;
}

static void
print_latency(uint64_t *d_elapsed, const SenderPhaseStats &stats,
              uint64_t num_iters, int gpu_id,
              size_t msg_size, bool use_warp, gpu_bench_op op)
{
    uint64_t h_elapsed = 0;
    cudaMemcpy(&h_elapsed, d_elapsed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[main] elapsed ticks (sender): %llu over %llu timed iters\n",
            (unsigned long long)h_elapsed, (unsigned long long)num_iters);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, gpu_id);
    double clock_hz = (double)clock_khz * 1000.0;
    fprintf(stderr, "[main] GPU SM clock: %.3f GHz\n", clock_hz / 1e9);

    double rtt_us     = (double)h_elapsed / (double)num_iters / clock_hz * 1e6;
    double one_way_us = rtt_us / 2.0;

    printf("op=%-12s  msg_size=%-6zu  iters=%-6llu  RTT=%.3f us  one-way=%.3f us  [%s]\n",
           op_to_string(op), msg_size, (unsigned long long)num_iters,
           rtt_us, one_way_us, use_warp ? "WARP" : "THREAD");
    printf("[pingpong-stats] meta op=%s msg_size=%zu iters=%llu warmup_skipped level=%s%s\n",
           op_to_string(op), msg_size, (unsigned long long)num_iters,
           use_warp ? "WARP" : "THREAD",
           op == gpu_bench_op::AtomicFlag
               ? " note=atomic-flag-uses-msg_size-as-counter-offset"
               : "");

    const PhaseSummary issue      = summarize_phase(stats.issue,      clock_hz);
    const PhaseSummary completion = summarize_phase(stats.completion, clock_hz);
    const PhaseSummary peer_wait  = summarize_phase(stats.peer_wait,  clock_hz);
    // one-way is just rtt with each sample halved; reuse the rtt buffer.
    PhaseStats one_way_phase = stats.rtt;
    one_way_phase.name = "one-way";
    const PhaseSummary one_way    = summarize_phase(one_way_phase,    clock_hz, /*scale=*/0.5);
    const PhaseSummary rtt        = summarize_phase(stats.rtt,        clock_hz);

    print_phase_summary_lines(issue);
    print_phase_summary_lines(completion);
    print_phase_summary_lines(peer_wait);
    print_phase_summary_lines(one_way);
    print_phase_summary_lines(rtt);

    // Non-overlapping lifecycle phases that should sum to ~rtt avg. Identical
    // for proxy and UCX device builds; the proxy worker's internal stage
    // acknowledgements are deliberately not polled from the GPU.
    std::vector<PhaseSummary> breakdown;
    breakdown.push_back(issue);
    breakdown.push_back(completion);
    breakdown.push_back(peer_wait);
    print_breakdown_chart(breakdown, rtt);
}

// ----------------------------------------------------------------------------
// Single-process benchmark (two threads, loopback TCP)
//
// Disabled in the CPU-proxy build: nixlProxyPublishContext writes a
// process-wide __device__ pointer, so two BenchContexts in one process would
// collide on it.
// ----------------------------------------------------------------------------
#ifndef NIXL_GPU_DEVICE_BACKEND_PROXY
static int
single_process_run(size_t msg_size, uint64_t num_iters, uint64_t warmup_iters,
                   int gpu_id, bool use_warp, int base_port, bool measure_submit,
                   gpu_bench_op op)
{
    fprintf(stderr,
            "[main] single-process mode  msg_size=%zu  iters=%llu  warmup=%llu"
            "  gpu=%d  level=%s  op=%s  ports=%d/%d\n",
            msg_size, (unsigned long long)num_iters, (unsigned long long)warmup_iters,
            gpu_id, use_warp ? "WARP" : "THREAD", op_to_string(op),
            base_port, base_port + 1);

    cudaSetDevice(gpu_id);
    uint64_t *d_elapsed_sender = nullptr, *d_elapsed_recvr = nullptr;
    if (cudaMalloc(&d_elapsed_sender, sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_elapsed_recvr,  sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[main] cudaMalloc d_elapsed failed\n");
        return 1;
    }
    cudaMemset(d_elapsed_sender, 0, sizeof(uint64_t));
    cudaMemset(d_elapsed_recvr,  0, sizeof(uint64_t));

    SenderPhaseStats sender_stats;
    if (!alloc_phase_stats(sender_stats.rtt, "rtt", num_iters, "main")) {
        cudaFree(d_elapsed_sender);
        cudaFree(d_elapsed_recvr);
        free_sender_phase_stats(sender_stats);
        return 1;
    }
    if (measure_submit) {
        if (!alloc_phase_stats(sender_stats.issue,      "issue",     num_iters, "main") ||
            !alloc_phase_stats(sender_stats.completion, "complete",  num_iters, "main") ||
            !alloc_phase_stats(sender_stats.peer_wait,  "peer-wait", num_iters, "main")) {
            cudaFree(d_elapsed_sender);
            cudaFree(d_elapsed_recvr);
            free_sender_phase_stats(sender_stats);
            return 1;
        }
    }

    BenchContext  sender_ctx, recvr_ctx;
    nixl_status_t sender_st = NIXL_SUCCESS, recvr_st = NIXL_SUCCESS;
    std::mutex setup_mutex;
    std::condition_variable setup_cv;
    int setup_ready_count = 0;
    bool setup_failed = false;

    auto wait_for_peer_setup = [&]() -> bool {
        std::unique_lock<std::mutex> lock(setup_mutex);
        ++setup_ready_count;
        setup_cv.notify_all();
        setup_cv.wait(lock, [&]() { return setup_ready_count == 2 || setup_failed; });
        return !setup_failed;
    };

    auto signal_setup_failed = [&]() {
        std::lock_guard<std::mutex> lock(setup_mutex);
        setup_failed = true;
        setup_cv.notify_all();
    };

    // ---- Sender thread -------------------------------------------------------
    std::thread sender_thr([&]() {
        fprintf(stderr, "[sender] thread started\n");

        BenchParams params;
        params.msg_size     = msg_size;
        params.num_iters    = num_iters;
        params.warmup_iters = warmup_iters;
        params.gpu_id       = gpu_id;
        params.is_sender    = true;
        params.op           = op;

        // Sender listens on base_port; receiver listens on base_port+1.
        sender_st = sender_ctx.setup(params, "127.0.0.1",
                                     /*peer_port=*/base_port + 1,
                                     /*my_port=*/base_port);
        if (sender_st != NIXL_SUCCESS) {
            fprintf(stderr, "[sender] setup failed (%d) — exiting thread\n", sender_st);
            signal_setup_failed();
            return;
        }
        fprintf(stderr, "[sender] setup complete\n");
        if (!wait_for_peer_setup()) return;

        gpu_bench_ctx kctx;
        kctx.local_mvh    = sender_ctx.local_mvh;
        kctx.remote_mvh   = sender_ctx.remote_mvh;
        kctx.send_buf     = (uint8_t *)sender_ctx.send_buf;
        kctx.recv_buf     = (uint8_t *)sender_ctx.recv_buf;
        kctx.msg_size     = msg_size;
        kctx.op           = op;
        kctx.num_iters    = num_iters;
        kctx.warmup_iters = warmup_iters;
        kctx.is_sender    = true;
        attach_phase_stats(kctx, sender_stats);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        fprintf(stderr, "[sender] launching %s kernel (%llu warmup + %llu timed iters)\n",
                use_warp ? "WARP" : "THREAD",
                (unsigned long long)warmup_iters, (unsigned long long)num_iters);

        if (use_warp) launch_pingpong_warp  (kctx, d_elapsed_sender, stream);
        else          launch_pingpong_thread(kctx, d_elapsed_sender, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        fprintf(stderr, "[sender] kernel finished\n");
    });

    // ---- Receiver thread -----------------------------------------------------
    std::thread recvr_thr([&]() {
        fprintf(stderr, "[recvr] thread started\n");

        BenchParams params;
        params.msg_size     = msg_size;
        params.num_iters    = num_iters;
        params.warmup_iters = warmup_iters;
        params.gpu_id       = gpu_id;
        params.is_sender    = false;
        params.op           = op;

        // Receiver listens on base_port+1; peer (sender) is on base_port.
        recvr_st = recvr_ctx.setup(params, "127.0.0.1",
                                   /*peer_port=*/base_port,
                                   /*my_port=*/base_port + 1);
        if (recvr_st != NIXL_SUCCESS) {
            fprintf(stderr, "[recvr] setup failed (%d) — exiting thread\n", recvr_st);
            signal_setup_failed();
            return;
        }
        fprintf(stderr, "[recvr] setup complete\n");
        if (!wait_for_peer_setup()) return;

        gpu_bench_ctx kctx;
        kctx.local_mvh    = recvr_ctx.local_mvh;
        kctx.remote_mvh   = recvr_ctx.remote_mvh;
        kctx.send_buf     = (uint8_t *)recvr_ctx.send_buf;
        kctx.recv_buf     = (uint8_t *)recvr_ctx.recv_buf;
        kctx.msg_size     = msg_size;
        kctx.op           = op;
        kctx.num_iters    = num_iters;
        kctx.warmup_iters = warmup_iters;
        kctx.is_sender    = false;
        kctx.issue_stats      = nullptr;
        kctx.completion_stats = nullptr;
        kctx.peer_wait_stats  = nullptr;
        kctx.rtt_stats        = nullptr;

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        fprintf(stderr, "[recvr] launching %s kernel\n", use_warp ? "WARP" : "THREAD");

        if (use_warp) launch_pingpong_warp  (kctx, d_elapsed_recvr, stream);
        else          launch_pingpong_thread(kctx, d_elapsed_recvr, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        fprintf(stderr, "[recvr] kernel finished\n");
    });

    sender_thr.join();
    recvr_thr.join();
    fprintf(stderr, "[main] both threads joined\n");

    if (sender_st != NIXL_SUCCESS || recvr_st != NIXL_SUCCESS) {
        fprintf(stderr, "[main] one or both sides failed — no latency output\n");
        cudaFree(d_elapsed_sender);
        cudaFree(d_elapsed_recvr);
        free_sender_phase_stats(sender_stats);
        return 1;
        // BenchContext destructors run here regardless
    }

    print_latency(d_elapsed_sender, sender_stats,
                  num_iters, gpu_id, msg_size, use_warp, op);
    cudaFree(d_elapsed_sender);
    cudaFree(d_elapsed_recvr);
    free_sender_phase_stats(sender_stats);
    fprintf(stderr, "[main] done\n");
    return 0;
    // sender_ctx and recvr_ctx destructors run here — NIXL teardown is automatic
}
#endif // !NIXL_GPU_DEVICE_BACKEND_PROXY

// ----------------------------------------------------------------------------
// Two-process benchmark (single-threaded; each process is one side)
// ----------------------------------------------------------------------------
static int
twoprocess_run(const char *peer_ip, int peer_port, int listen_port,
               size_t msg_size, uint64_t num_iters, uint64_t warmup_iters,
               int gpu_id, bool use_warp, bool is_sender, bool measure_submit,
               gpu_bench_op op)
{
    const char *role = is_sender ? "sender" : "receiver";
    fprintf(stderr,
            "[main] two-process mode  role=%s  peer=%s:%d  listen=%d"
            "  op=%s  msg_size=%zu  level=%s\n",
            role, peer_ip, peer_port, listen_port, op_to_string(op),
            msg_size, use_warp ? "WARP" : "THREAD");

    BenchParams params;
    params.msg_size     = msg_size;
    params.num_iters    = num_iters;
    params.warmup_iters = warmup_iters;
    params.gpu_id       = gpu_id;
    params.is_sender    = is_sender;
    params.op           = op;

    BenchContext ctx;
    nixl_status_t st = ctx.setup(params, peer_ip, peer_port, listen_port);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] setup failed: %d\n", role, st);
        return 1;
    }

    cudaSetDevice(gpu_id);
    uint64_t *d_elapsed = nullptr;
    if (cudaMalloc(&d_elapsed, sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc d_elapsed failed\n", role);
        return 1;
    }
    cudaMemset(d_elapsed, 0, sizeof(uint64_t));

    SenderPhaseStats sender_stats;
    auto bail = [&]() {
        cudaFree(d_elapsed);
        free_sender_phase_stats(sender_stats);
        return 1;
    };
    if (is_sender) {
        if (!alloc_phase_stats(sender_stats.rtt, "rtt", num_iters, role)) {
            return bail();
        }
    }
    if (measure_submit && is_sender) {
        if (!alloc_phase_stats(sender_stats.issue,      "issue",     num_iters, role) ||
            !alloc_phase_stats(sender_stats.completion, "complete",  num_iters, role) ||
            !alloc_phase_stats(sender_stats.peer_wait,  "peer-wait", num_iters, role)) {
            return bail();
        }
    }

    gpu_bench_ctx kctx;
    kctx.local_mvh    = ctx.local_mvh;
    kctx.remote_mvh   = ctx.remote_mvh;
    kctx.send_buf     = (uint8_t *)ctx.send_buf;
    kctx.recv_buf     = (uint8_t *)ctx.recv_buf;
    kctx.msg_size     = msg_size;
    kctx.op           = op;
    kctx.num_iters    = num_iters;
    kctx.warmup_iters = warmup_iters;
    kctx.is_sender    = is_sender;
    attach_phase_stats(kctx, sender_stats);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    fprintf(stderr, "[%s] launching %s kernel (%llu warmup + %llu timed iters)\n",
            role, use_warp ? "WARP" : "THREAD",
            (unsigned long long)warmup_iters, (unsigned long long)num_iters);

    if (use_warp) launch_pingpong_warp  (kctx, d_elapsed, stream);
    else          launch_pingpong_thread(kctx, d_elapsed, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    fprintf(stderr, "[%s] kernel finished\n", role);

    if (is_sender)
        print_latency(d_elapsed, sender_stats,
                      num_iters, gpu_id, msg_size, use_warp, op);

    cudaFree(d_elapsed);
    free_sender_phase_stats(sender_stats);
    fprintf(stderr, "[main] done\n");
    return 0;
    // ctx destructor runs here — NIXL teardown is automatic
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int
main(int argc, char *argv[]) {
    const char *role_str  = nullptr;
    const char *peer_ip   = nullptr;
    int  peer_port        = 0;
    int  listen_port      = 0;
#ifndef NIXL_GPU_DEVICE_BACKEND_PROXY
    int  base_port        = 12300;
#endif
    size_t   msg_size     = 8;
    uint64_t num_iters    = 1000;
    uint64_t warmup_iters = 100;
    int      gpu_id       = 0;
    bool     use_warp     = false;
    bool     measure_submit = true;
    bool     single_process = false;
    gpu_bench_op op = gpu_bench_op::Put;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--role")         && i + 1 < argc) role_str    = argv[++i];
        else if (!strcmp(argv[i], "--peer-ip")      && i + 1 < argc) peer_ip     = argv[++i];
        else if (!strcmp(argv[i], "--peer-port")    && i + 1 < argc) peer_port   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--listen-port")  && i + 1 < argc) listen_port = atoi(argv[++i]);
#ifndef NIXL_GPU_DEVICE_BACKEND_PROXY
        else if (!strcmp(argv[i], "--base-port")    && i + 1 < argc) base_port   = atoi(argv[++i]);
#endif
        else if (!strcmp(argv[i], "--msg-size")     && i + 1 < argc) msg_size    = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--iters")        && i + 1 < argc) num_iters   = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")       && i + 1 < argc) warmup_iters = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")          && i + 1 < argc) gpu_id      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warp"))                          use_warp    = true;
        else if (!strcmp(argv[i], "--op")           && i + 1 < argc) {
            if (!parse_bench_op(argv[++i], op)) {
                fprintf(stderr, "Unknown op '%s'; expected put or atomic-flag\n", argv[i]);
                usage(argv[0]);
            }
        }
        else if (!strcmp(argv[i], "--measure-submit"))                measure_submit = true;
        else if (!strcmp(argv[i], "--no-measure-submit"))             measure_submit = false;
        else if (!strcmp(argv[i], "--single-process"))                single_process = true;
        else usage(argv[0]);
    }

    // Default to single-process when no two-process args given
    if (!role_str && !peer_ip && peer_port == 0 && listen_port == 0) single_process = true;

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    if (single_process) {
        fprintf(stderr,
                "Single-process mode is not supported in the CPU-proxy build; "
                "use --role sender/--role receiver across two processes.\n");
        usage(argv[0]);
    }
#else
    if (single_process) {
        if (role_str || peer_ip || peer_port || listen_port) {
            fprintf(stderr, "--single-process is mutually exclusive with "
                            "--role/--peer-ip/--peer-port/--listen-port\n");
            usage(argv[0]);
        }
        return single_process_run(msg_size, num_iters, warmup_iters, gpu_id, use_warp,
                                  base_port, measure_submit, op);
    }
#endif

    // Two-process mode
    if (!role_str || !peer_ip || peer_port == 0 || listen_port == 0) usage(argv[0]);

    bool is_sender = (strcmp(role_str, "sender") == 0);
    if (!is_sender && strcmp(role_str, "receiver") != 0) {
        fprintf(stderr, "Unknown role '%s'; expected sender or receiver\n", role_str);
        usage(argv[0]);
    }

    return twoprocess_run(peer_ip, peer_port, listen_port,
                          msg_size, num_iters, warmup_iters,
                          gpu_id, use_warp, is_sender, measure_submit, op);
}
