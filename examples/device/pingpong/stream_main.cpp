// Windowed one-directional streaming BANDWIDTH bench (see stream_bench.h).
//
// Reuses BenchContext (bench_host.cpp) for ALL the RDMA setup + (proxy build)
// device-context publish — identical to the latency bench. The sender launches
// the windowed streaming kernel; the receiver is a passive RDMA-write target
// (no data-path kernel) that just waits for a host "done" notification.
//
// Output (sender only), one line, easy to parse:
//   [stream] op=put level=THREAD msg_size=65536 window=16 warps=1 channels=1
//            iters=20000 secs=0.123 GB/s=42.10 Mops/s=0.64

#include "bench_host.h"
#include "stream_bench.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

static const char *
op_to_string(gpu_bench_op op) {
    return op == gpu_bench_op::Put ? "put" : "atomic-flag";
}

static bool
parse_op(const char *s, gpu_bench_op &op) {
    if (!strcmp(s, "put"))                                       op = gpu_bench_op::Put;
    else if (!strcmp(s, "atomic-flag") || !strcmp(s, "atomic"))  op = gpu_bench_op::AtomicFlag;
    else return false;
    return true;
}

static void
usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --role sender|receiver --peer-ip IP --peer-port P --listen-port L\n"
        "          [--msg-size N] [--iters N] [--warmup N] [--window W]\n"
        "          [--warps P] [--channels C] [--gpu G] [--warp] [--op put|atomic-flag]\n"
        "\n"
        "One-directional windowed streaming bandwidth: keep W puts in flight per\n"
        "warp across P warps; sustained throughput = warps*iters*msg_size / wall.\n"
        "channels selects QP fan-out (direct); the proxy build forces channels=1.\n",
        prog);
    exit(1);
}

int
main(int argc, char *argv[]) {
    const char *role_str = nullptr;
    const char *peer_ip  = nullptr;
    int peer_port = 0, listen_port = 0, gpu_id = 0;
    size_t   msg_size = 65536;
    uint64_t num_iters = 20000, warmup_iters = 2000;
    uint32_t window = 16, warps = 1, channels = 1;
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
        else if (!strcmp(argv[i], "--window")      && i + 1 < argc) window      = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warps")       && i + 1 < argc) warps       = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--channels")    && i + 1 < argc) channels    = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")         && i + 1 < argc) gpu_id      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warp"))                        use_warp    = true;
        else if (!strcmp(argv[i], "--op")          && i + 1 < argc) {
            if (!parse_op(argv[++i], op)) { fprintf(stderr, "bad --op\n"); usage(argv[0]); }
        }
        else usage(argv[0]);
    }

    if (!role_str || !peer_ip || peer_port == 0 || listen_port == 0) usage(argv[0]);
    const bool is_sender = !strcmp(role_str, "sender");
    if (!is_sender && strcmp(role_str, "receiver")) usage(argv[0]);
    if (window == 0) window = 1;
    if (warps  == 0) warps  = 1;

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    // The proxy demo runs a single channel / single worker by construction
    // (bench_host.cpp). Every stream funnels into that one ring on purpose.
    if (channels != 1) {
        fprintf(stderr, "[stream] proxy build forces channels=1 (was %u)\n", channels);
        channels = 1;
    }
#endif

    const char *role = is_sender ? "sender" : "receiver";
    fprintf(stderr,
        "[stream] role=%s peer=%s:%d listen=%d op=%s msg_size=%zu level=%s "
        "iters=%llu warmup=%llu window=%u warps=%u channels=%u\n",
        role, peer_ip, peer_port, listen_port, op_to_string(op), msg_size,
        use_warp ? "WARP" : "THREAD",
        (unsigned long long)num_iters, (unsigned long long)warmup_iters,
        window, warps, channels);

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

    const std::string peer_name = is_sender ? "receiver" : "sender";
    nixl_opt_args_t notif_args;
    notif_args.backends.push_back(ctx.ucx_backend);

    if (!is_sender) {
        // Passive RDMA target: no data-path kernel. Keep the agent (and thus the
        // RC connection + registered recv_buf) alive until the sender signals it
        // has drained every put, then exit cleanly.
        fprintf(stderr, "[receiver] streaming target ready, waiting for done...\n");
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(600);
        bool done = false;
        while (!done) {
            nixl_notifs_t notifs;
            st = ctx.agent->getNotifs(notifs, &notif_args);
            if (st == NIXL_SUCCESS) {
                auto it = notifs.find(peer_name);
                if (it != notifs.end() && !it->second.empty()) {
                    done = true;
                    break;
                }
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                fprintf(stderr, "[receiver] timed out waiting for done\n");
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        fprintf(stderr, "[receiver] done\n");
        return 0;
    }

    // ---- sender ----
    cudaSetDevice(gpu_id);
    uint64_t *d_elapsed = nullptr;
    void *d_reqs = nullptr;
    const size_t reqs_count = (size_t)warps * window;
    const size_t reqs_bytes = stream_xfer_status_bytes() * reqs_count;
    if (cudaMalloc(&d_elapsed, sizeof(uint64_t) * warps) != cudaSuccess ||
        cudaMalloc(&d_reqs, reqs_bytes) != cudaSuccess) {
        fprintf(stderr, "[sender] cudaMalloc failed\n");
        return 1;
    }
    cudaMemset(d_elapsed, 0, sizeof(uint64_t) * warps);
    cudaMemset(d_reqs, 0, reqs_bytes);

    stream_bench_ctx kctx;
    kctx.local_mvh     = ctx.local_mvh;
    kctx.remote_mvh    = ctx.remote_mvh;
    kctx.send_buf      = (uint8_t *)ctx.send_buf;
    kctx.recv_buf      = (uint8_t *)ctx.recv_buf;
    kctx.msg_size      = msg_size;
    kctx.op            = op;
    kctx.num_iters     = num_iters;
    kctx.warmup_iters  = warmup_iters;
    kctx.window        = window;
    kctx.num_warps     = warps;
    kctx.num_channels  = channels;
    kctx.reqs          = static_cast<nixlGpuXferStatusH *>(d_reqs);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    fprintf(stderr, "[sender] launching %s streaming kernel (%u warps)\n",
            use_warp ? "WARP" : "THREAD", warps);
    if (use_warp) launch_stream_warp  (kctx, d_elapsed, stream);
    else          launch_stream_thread(kctx, d_elapsed, stream);
    cudaError_t kerr = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[sender] kernel error: %s\n", cudaGetErrorString(kerr));
        cudaFree(d_elapsed); cudaFree(d_reqs);
        return 1;
    }
    fprintf(stderr, "[sender] kernel finished\n");

    // Signal the receiver it may tear down (all puts have drained).
    nixl_blob_t done_blob("DONE");
    while ((st = ctx.agent->genNotif(peer_name, done_blob, &notif_args)) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    std::vector<uint64_t> elapsed(warps, 0);
    cudaMemcpy(elapsed.data(), d_elapsed, sizeof(uint64_t) * warps,
               cudaMemcpyDeviceToHost);
    uint64_t max_ticks = 0, min_ticks = ~0ull;
    for (uint32_t w = 0; w < warps; w++) {
        if (elapsed[w] > max_ticks) max_ticks = elapsed[w];
        if (elapsed[w] < min_ticks) min_ticks = elapsed[w];
    }

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, gpu_id);
    const double clock_hz = (double)clock_khz * 1000.0;
    const double wall_s   = (double)max_ticks / clock_hz;       // slowest warp = wall
    const double total_bytes = (double)warps * (double)num_iters * (double)msg_size;
    const double total_ops   = (double)warps * (double)num_iters;
    const double gbps  = wall_s > 0 ? total_bytes / wall_s / 1e9 : 0.0;
    const double mops  = wall_s > 0 ? total_ops   / wall_s / 1e6 : 0.0;
    const double skew  = max_ticks > 0 ? (double)(max_ticks - min_ticks) / (double)max_ticks : 0.0;

    fprintf(stderr,
            "[stream] GPU SM clock: %.3f GHz (max_ticks=%llu min_ticks=%llu skew=%.1f%%)\n",
            clock_hz / 1e9, (unsigned long long)max_ticks,
            (unsigned long long)min_ticks, skew * 100.0);
    printf("[stream] op=%s level=%s msg_size=%zu window=%u warps=%u channels=%u "
           "iters=%llu secs=%.6f GB/s=%.3f Mops/s=%.4f\n",
           op_to_string(op), use_warp ? "WARP" : "THREAD", msg_size,
           window, warps, channels, (unsigned long long)num_iters,
           wall_s, gbps, mops);

    cudaFree(d_elapsed);
    cudaFree(d_reqs);
    fprintf(stderr, "[sender] done\n");
    return 0;
}
