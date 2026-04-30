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
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

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
            "    [--no-measure-submit]\n"
            "    [--base-port <port>]   loopback listen ports (default 12300);\n"
            "                           sender uses base, receiver uses base+1\n",
            prog, prog);
#endif
    exit(1);
}

// ----------------------------------------------------------------------------
// Shared helper: print latency from sender's elapsed tick counter
// ----------------------------------------------------------------------------
struct TimingStatsUs {
    double avg = -1.0;
    double min = -1.0;
    double max = -1.0;
    double stddev = -1.0;
    uint64_t count = 0;
};

static TimingStatsUs
read_cycle_stats_us(gpu_cycle_stats *d_stats, double clock_hz, double scale = 1.0)
{
    TimingStatsUs out;
    if (d_stats == nullptr) {
        return out;
    }

    gpu_cycle_stats h_stats{};
    cudaMemcpy(&h_stats, d_stats, sizeof(gpu_cycle_stats), cudaMemcpyDeviceToHost);
    if (h_stats.count == 0) {
        return out;
    }

    const double count = static_cast<double>(h_stats.count);
    const double avg_cycles = static_cast<double>(h_stats.sum) / count;
    const double mean_sq_cycles = h_stats.sum_sq / count;
    const double variance_cycles = std::fmax(0.0, mean_sq_cycles - avg_cycles * avg_cycles);
    const double cycles_to_us = 1e6 / clock_hz * scale;

    out.avg = avg_cycles * cycles_to_us;
    out.min = static_cast<double>(h_stats.min) * cycles_to_us;
    out.max = static_cast<double>(h_stats.max) * cycles_to_us;
    out.stddev = std::sqrt(variance_cycles) * cycles_to_us;
    out.count = h_stats.count;
    return out;
}

static void
print_timing_row(const char *name, const TimingStatsUs &stats)
{
    printf("  %-8s %14.6f  %14.6f  %14.6f  %14.6f\n",
           name, stats.avg, stats.min, stats.max, stats.stddev);
}

static void
print_latency(uint64_t *d_elapsed, gpu_cycle_stats *d_issue_stats,
              gpu_cycle_stats *d_submit_stats, gpu_cycle_stats *d_rtt_stats,
              uint64_t num_iters, int gpu_id,
              size_t msg_size, bool use_warp)
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
    TimingStatsUs issue = read_cycle_stats_us(d_issue_stats, clock_hz);
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    TimingStatsUs submit = read_cycle_stats_us(d_submit_stats, clock_hz);
#endif
    TimingStatsUs rtt = read_cycle_stats_us(d_rtt_stats, clock_hz);
    TimingStatsUs one_way = read_cycle_stats_us(d_rtt_stats, clock_hz, 0.5);

    printf("msg_size=%-6zu  iters=%-6llu  RTT=%.3f us  one-way=%.3f us  [%s]\n",
           msg_size, (unsigned long long)num_iters, rtt_us, one_way_us,
           use_warp ? "WARP" : "THREAD");
    printf("metrics:\n");
    printf("  msg_size=%zu  iters=%llu  level=%s  samples=%llu\n",
           msg_size, (unsigned long long)num_iters, use_warp ? "WARP" : "THREAD",
           (unsigned long long)rtt.count);
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    printf("  one-way includes: issue + submit + backend progress/completion + network/peer response\n");
#else
    printf("  one-way includes: issue/submit + backend progress/completion + network/peer response\n");
#endif
    printf("  %-8s %14s  %14s  %14s  %14s\n",
           "", "avg_us", "min_us", "max_us", "stddev_us");
    print_timing_row("issue", issue);
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    print_timing_row("submit", submit);
#endif
    print_timing_row("one-way", one_way);
    print_timing_row("rtt", rtt);
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
                   int gpu_id, bool use_warp, int base_port, bool measure_submit)
{
    fprintf(stderr,
            "[main] single-process mode  msg_size=%zu  iters=%llu  warmup=%llu"
            "  gpu=%d  level=%s  ports=%d/%d\n",
            msg_size, (unsigned long long)num_iters, (unsigned long long)warmup_iters,
            gpu_id, use_warp ? "WARP" : "THREAD", base_port, base_port + 1);

    cudaSetDevice(gpu_id);
    uint64_t *d_elapsed_sender = nullptr, *d_elapsed_recvr = nullptr;
    gpu_cycle_stats *d_issue_sender = nullptr, *d_submit_sender = nullptr, *d_rtt_sender = nullptr;
    if (cudaMalloc(&d_elapsed_sender, sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_elapsed_recvr,  sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[main] cudaMalloc d_elapsed failed\n");
        return 1;
    }
    cudaMemset(d_elapsed_sender, 0, sizeof(uint64_t));
    cudaMemset(d_elapsed_recvr,  0, sizeof(uint64_t));
    if (cudaMalloc(&d_rtt_sender, sizeof(gpu_cycle_stats)) != cudaSuccess) {
        fprintf(stderr, "[main] cudaMalloc RTT stats failed\n");
        cudaFree(d_elapsed_sender);
        cudaFree(d_elapsed_recvr);
        return 1;
    }
    cudaMemset(d_rtt_sender, 0, sizeof(gpu_cycle_stats));
    if (measure_submit) {
        if (cudaMalloc(&d_issue_sender, sizeof(gpu_cycle_stats)) != cudaSuccess ||
            cudaMalloc(&d_submit_sender, sizeof(gpu_cycle_stats)) != cudaSuccess) {
            fprintf(stderr, "[main] cudaMalloc timing counters failed\n");
            cudaFree(d_elapsed_sender);
            cudaFree(d_elapsed_recvr);
            cudaFree(d_issue_sender);
            cudaFree(d_submit_sender);
            cudaFree(d_rtt_sender);
            return 1;
        }
        cudaMemset(d_issue_sender, 0, sizeof(gpu_cycle_stats));
        cudaMemset(d_submit_sender, 0, sizeof(gpu_cycle_stats));
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
        kctx.num_iters    = num_iters;
        kctx.warmup_iters = warmup_iters;
        kctx.is_sender    = true;
        kctx.issue_stats = d_issue_sender;
        kctx.submit_stats = d_submit_sender;
        kctx.rtt_stats = d_rtt_sender;

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
        kctx.num_iters    = num_iters;
        kctx.warmup_iters = warmup_iters;
        kctx.is_sender    = false;
        kctx.issue_stats = nullptr;
        kctx.submit_stats = nullptr;
        kctx.rtt_stats = nullptr;

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
        cudaFree(d_issue_sender);
        cudaFree(d_submit_sender);
        cudaFree(d_rtt_sender);
        return 1;
        // BenchContext destructors run here regardless
    }

    print_latency(d_elapsed_sender, d_issue_sender, d_submit_sender, d_rtt_sender,
                  num_iters, gpu_id, msg_size, use_warp);
    cudaFree(d_elapsed_sender);
    cudaFree(d_elapsed_recvr);
    cudaFree(d_issue_sender);
    cudaFree(d_submit_sender);
    cudaFree(d_rtt_sender);
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
               int gpu_id, bool use_warp, bool is_sender, bool measure_submit)
{
    const char *role = is_sender ? "sender" : "receiver";
    fprintf(stderr,
            "[main] two-process mode  role=%s  peer=%s:%d  listen=%d"
            "  msg_size=%zu  level=%s\n",
            role, peer_ip, peer_port, listen_port, msg_size,
            use_warp ? "WARP" : "THREAD");

    BenchParams params;
    params.msg_size     = msg_size;
    params.num_iters    = num_iters;
    params.warmup_iters = warmup_iters;
    params.gpu_id       = gpu_id;
    params.is_sender    = is_sender;

    BenchContext ctx;
    nixl_status_t st = ctx.setup(params, peer_ip, peer_port, listen_port);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] setup failed: %d\n", role, st);
        return 1;
    }

    cudaSetDevice(gpu_id);
    uint64_t *d_elapsed = nullptr;
    gpu_cycle_stats *d_issue_stats = nullptr;
    gpu_cycle_stats *d_submit_stats = nullptr;
    gpu_cycle_stats *d_rtt_stats = nullptr;
    if (cudaMalloc(&d_elapsed, sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc d_elapsed failed\n", role);
        return 1;
    }
    cudaMemset(d_elapsed, 0, sizeof(uint64_t));
    if (is_sender) {
        if (cudaMalloc(&d_rtt_stats, sizeof(gpu_cycle_stats)) != cudaSuccess) {
            fprintf(stderr, "[%s] cudaMalloc RTT stats failed\n", role);
            cudaFree(d_elapsed);
            return 1;
        }
        cudaMemset(d_rtt_stats, 0, sizeof(gpu_cycle_stats));
    }
    if (measure_submit && is_sender) {
        if (cudaMalloc(&d_issue_stats, sizeof(gpu_cycle_stats)) != cudaSuccess ||
            cudaMalloc(&d_submit_stats, sizeof(gpu_cycle_stats)) != cudaSuccess) {
            fprintf(stderr, "[%s] cudaMalloc timing counters failed\n", role);
            cudaFree(d_elapsed);
            cudaFree(d_issue_stats);
            cudaFree(d_submit_stats);
            cudaFree(d_rtt_stats);
            return 1;
        }
        cudaMemset(d_issue_stats, 0, sizeof(gpu_cycle_stats));
        cudaMemset(d_submit_stats, 0, sizeof(gpu_cycle_stats));
    }

    gpu_bench_ctx kctx;
    kctx.local_mvh    = ctx.local_mvh;
    kctx.remote_mvh   = ctx.remote_mvh;
    kctx.send_buf     = (uint8_t *)ctx.send_buf;
    kctx.recv_buf     = (uint8_t *)ctx.recv_buf;
    kctx.msg_size     = msg_size;
    kctx.num_iters    = num_iters;
    kctx.warmup_iters = warmup_iters;
    kctx.is_sender    = is_sender;
    kctx.issue_stats = d_issue_stats;
    kctx.submit_stats = d_submit_stats;
    kctx.rtt_stats = d_rtt_stats;

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
        print_latency(d_elapsed, d_issue_stats, d_submit_stats, d_rtt_stats,
                      num_iters, gpu_id, msg_size, use_warp);

    cudaFree(d_elapsed);
    cudaFree(d_issue_stats);
    cudaFree(d_submit_stats);
    cudaFree(d_rtt_stats);
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
                                  base_port, measure_submit);
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
                          gpu_id, use_warp, is_sender, measure_submit);
}
