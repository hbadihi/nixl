// Minimal two-process ping-pong latency bench (see simple_bench.h).
//
// Reuses BenchContext (bench_host.cpp) for ALL the RDMA setup + (proxy build)
// device-context publish. The only thing this file adds is: alloc two device
// scalars, launch the minimal kernel, convert clock64 ticks to microseconds.
//
// Output (sender only), one line, easy to parse:
//   [simple] op=put level=THREAD msg_size=512 iters=5000 issue=1.389 us RTT=11.167 us one-way=5.583 us

#include "bench_host.h"
#include "simple_bench.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static const char *
op_to_string(gpu_bench_op op) {
    return op == gpu_bench_op::Put ? "put" : "atomic-flag";
}

static bool
parse_op(const char *s, gpu_bench_op &op) {
    if (!strcmp(s, "put"))                                op = gpu_bench_op::Put;
    else if (!strcmp(s, "atomic-flag") || !strcmp(s, "atomic")) op = gpu_bench_op::AtomicFlag;
    else return false;
    return true;
}

static void
usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --role sender|receiver --peer-ip IP --peer-port P --listen-port L\n"
        "          [--msg-size N] [--iters N] [--warmup N] [--gpu G] [--warp]\n"
        "          [--op put|atomic-flag]\n"
        "\n"
        "Minimal ping-pong: times only the nixlPut/nixlAtomicAdd call (issue) and\n"
        "the full RTT loop; one-way = RTT/2. No completion/stage polling.\n", prog);
    exit(1);
}

int
main(int argc, char *argv[]) {
    const char *role_str = nullptr;
    const char *peer_ip  = nullptr;
    int peer_port = 0, listen_port = 0, gpu_id = 0;
    size_t   msg_size = 8;
    uint64_t num_iters = 1000, warmup_iters = 100;
    bool use_warp = false;
    gpu_bench_op op = gpu_bench_op::Put;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--role")        && i + 1 < argc) role_str    = argv[++i];
        else if (!strcmp(argv[i], "--peer-ip")     && i + 1 < argc) peer_ip     = argv[++i];
        else if (!strcmp(argv[i], "--peer-port")   && i + 1 < argc) peer_port   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--listen-port") && i + 1 < argc) listen_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--msg-size")    && i + 1 < argc) msg_size    = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--iters")       && i + 1 < argc) num_iters   = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")      && i + 1 < argc) warmup_iters = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")         && i + 1 < argc) gpu_id      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warp"))                         use_warp    = true;
        else if (!strcmp(argv[i], "--op")          && i + 1 < argc) {
            if (!parse_op(argv[++i], op)) { fprintf(stderr, "bad --op\n"); usage(argv[0]); }
        }
        else usage(argv[0]);
    }

    if (!role_str || !peer_ip || peer_port == 0 || listen_port == 0) usage(argv[0]);
    const bool is_sender = !strcmp(role_str, "sender");
    if (!is_sender && strcmp(role_str, "receiver")) usage(argv[0]);

    const char *role = is_sender ? "sender" : "receiver";
    fprintf(stderr,
        "[simple] role=%s peer=%s:%d listen=%d op=%s msg_size=%zu level=%s iters=%llu warmup=%llu\n",
        role, peer_ip, peer_port, listen_port, op_to_string(op), msg_size,
        use_warp ? "WARP" : "THREAD",
        (unsigned long long)num_iters, (unsigned long long)warmup_iters);

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
    uint64_t *d_elapsed = nullptr, *d_issue = nullptr;
    if (cudaMalloc(&d_elapsed, sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_issue,   sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc failed\n", role);
        return 1;
    }
    cudaMemset(d_elapsed, 0, sizeof(uint64_t));
    cudaMemset(d_issue,   0, sizeof(uint64_t));

    simple_bench_ctx kctx;
    kctx.local_mvh    = ctx.local_mvh;
    kctx.remote_mvh   = ctx.remote_mvh;
    kctx.send_buf     = (uint8_t *)ctx.send_buf;
    kctx.recv_buf     = (uint8_t *)ctx.recv_buf;
    kctx.msg_size     = msg_size;
    kctx.op           = op;
    kctx.num_iters    = num_iters;
    kctx.warmup_iters = warmup_iters;
    kctx.is_sender    = is_sender;
    kctx.issue_ticks  = is_sender ? d_issue : nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    fprintf(stderr, "[%s] launching %s kernel\n", role, use_warp ? "WARP" : "THREAD");
    if (use_warp) launch_simple_warp  (kctx, d_elapsed, stream);
    else          launch_simple_thread(kctx, d_elapsed, stream);
    cudaError_t kerr = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[%s] kernel error: %s\n", role, cudaGetErrorString(kerr));
        cudaFree(d_elapsed); cudaFree(d_issue);
        return 1;
    }
    fprintf(stderr, "[%s] kernel finished\n", role);

    if (is_sender) {
        uint64_t elapsed_ticks = 0, issue_ticks = 0;
        cudaMemcpy(&elapsed_ticks, d_elapsed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&issue_ticks,   d_issue,   sizeof(uint64_t), cudaMemcpyDeviceToHost);

        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, gpu_id);
        const double clock_hz = (double)clock_khz * 1000.0;

        const double rtt_us     = (double)elapsed_ticks / (double)num_iters / clock_hz * 1e6;
        const double one_way_us = rtt_us / 2.0;
        const double issue_us   = (double)issue_ticks / (double)num_iters / clock_hz * 1e6;

        fprintf(stderr, "[simple] GPU SM clock: %.3f GHz (elapsed=%llu issue=%llu ticks over %llu iters)\n",
                clock_hz / 1e9, (unsigned long long)elapsed_ticks,
                (unsigned long long)issue_ticks, (unsigned long long)num_iters);
        printf("[simple] op=%s level=%s msg_size=%zu iters=%llu "
               "issue=%.3f us RTT=%.3f us one-way=%.3f us\n",
               op_to_string(op), use_warp ? "WARP" : "THREAD", msg_size,
               (unsigned long long)num_iters, issue_us, rtt_us, one_way_us);
    }

    cudaFree(d_elapsed);
    cudaFree(d_issue);
    fprintf(stderr, "[%s] done\n", role);
    return 0;
}
