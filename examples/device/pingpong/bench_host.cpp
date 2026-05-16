#include "bench_host.h"
#include "bench_kernel_iface.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

namespace {
struct PeerRecvInfo {
    uintptr_t recv_addr;
    uint64_t gpu_id;
};

template <typename DList>
nixl_status_t
prep_memview_with_retries(nixlAgent *agent,
                          const DList &dlist,
                          nixlMemViewH &mvh,
                          const std::string &agent_name,
                          const char *label)
{
    nixl_status_t st = NIXL_SUCCESS;
    for (int attempt = 0; attempt < 5; ++attempt) {
        st = agent->prepMemView(dlist, mvh);
        if (st == NIXL_SUCCESS) {
            return st;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    fprintf(stderr, "[%s] prepMemView(%s) failed after retries: %d\n",
            agent_name.c_str(), label, st);
    return st;
}
} // namespace

// ---- BenchContext::setup -----------------------------------------------------

nixl_status_t
BenchContext::setup(const BenchParams &params,
                    const char *peer_ip, int peer_port, int my_port)
{
    is_sender = params.is_sender;
    gpu_id    = params.gpu_id;
    buf_size  = params.msg_size + sizeof(uint64_t);

    const std::string my_name   = params.is_sender ? "sender"   : "receiver";
    const std::string peer_name = params.is_sender ? "receiver" : "sender";

    // 1. Bind CUDA device
    cudaSetDevice(params.gpu_id);

    // 2. Create NIXL agent with both a progress thread (drives UCX completions)
    //    and a listen thread (accepts incoming TCP metadata connections from peer).
    //    The constructor throws std::runtime_error if the listen port is in use.
    nixlAgentConfig cfg(/*useProgThread=*/true, /*useListenThread=*/true, my_port);
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    // Route GPU-issued nixlPut through the CPU proxy runtime. One worker /
    // one channel keeps the demo on the currently validated UCX proxy path;
    // the shared UCX postXfer path still needs multi-worker validation.
    cfg.enableDeviceProxy = true;
    cfg.proxyChannelCount = 1;
    cfg.proxyWorkerCount  = 1;
#endif
    try {
        agent = std::make_unique<nixlAgent>(my_name, cfg);
    } catch (const std::exception &e) {
        fprintf(stderr, "[%s] nixlAgent construction failed (port %d in use?): %s\n",
                my_name.c_str(), my_port, e.what());
        return NIXL_ERR_NOT_FOUND;
    }

    // 3. Create UCX backend
    nixl_b_params_t bparams;
    bparams["ucx_error_handling_mode"] = "none";
    nixl_status_t st = agent->createBackend("UCX", bparams, ucx_backend);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] createBackend failed: %d\n", my_name.c_str(), st);
        return st;
    }

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    // The proxy runtime is created lazily by the agent during createBackend().
    // Publish its device context to the process-wide __device__ pointer that
    // load_proxy_context() reads from inside the kernel.
    void *proxy_ctx = agent->getProxyDeviceContext();
    if (proxy_ctx == nullptr) {
        fprintf(stderr, "[%s] proxy device context not available\n", my_name.c_str());
        return NIXL_ERR_BACKEND;
    }
    if (bench_proxy_publish_context(proxy_ctx) != cudaSuccess) {
        fprintf(stderr, "[%s] bench_proxy_publish_context failed\n", my_name.c_str());
        return NIXL_ERR_BACKEND;
    }
#endif

    // 4. Allocate and zero device buffers.
    if (cudaMalloc(&send_buf, buf_size) != cudaSuccess ||
        cudaMalloc(&recv_buf, buf_size) != cudaSuccess) {
        if (send_buf) {
            cudaFree(send_buf);
            send_buf = nullptr;
        }
        fprintf(stderr, "[%s] cudaMalloc failed\n", my_name.c_str());
        return NIXL_ERR_NOT_FOUND;
    }
    cudaMemset(send_buf, 0, buf_size);
    cudaMemset(recv_buf, 0, buf_size);

    // 5. Register both buffers.
    //    recv_buf must be registered so its UCX rkey is serialised into the
    //    metadata blob; the peer needs that rkey to PUT into our recv_buf.
    nixl_reg_dlist_t send_dlist(VRAM_SEG), recv_dlist(VRAM_SEG);
    send_dlist.addDesc(nixlBlobDesc((uintptr_t)send_buf, buf_size, params.gpu_id, ""));
    recv_dlist.addDesc(nixlBlobDesc((uintptr_t)recv_buf, buf_size, params.gpu_id, ""));

    st = agent->registerMem(send_dlist);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] registerMem(send) failed: %d\n", my_name.c_str(), st);
        return st;
    }
    st = agent->registerMem(recv_dlist);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] registerMem(recv) failed: %d\n", my_name.c_str(), st);
        return st;
    }

    // 6. Metadata exchange via TCP.
    //
    //    The metadata blob contains UCX connection info and rkeys for all
    //    registered buffers. fetchRemoteMD connects to the peer's listen port
    //    and downloads the peer's blob; sendLocalMD connects and uploads ours.
    //
    //    Sender drives both calls; receiver's listen thread handles them
    //    passively — no explicit metadata API calls on the receiver side.
    nixl_opt_args_t md_args;
    md_args.ipAddr = peer_ip;
    md_args.port   = peer_port;   // peer's listen port

    if (params.is_sender) {
        fprintf(stderr, "[%s] fetching remote MD from %s:%d ...\n",
                my_name.c_str(), peer_ip, peer_port);
        // Retry: receiver's listen thread may not be ready yet.
        while ((st = agent->fetchRemoteMD(peer_name, &md_args)) != NIXL_SUCCESS) {
            fprintf(stderr, "[%s] fetchRemoteMD not ready (%d), retrying...\n",
                    my_name.c_str(), st);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        fprintf(stderr, "[%s] fetchRemoteMD done, pushing local MD...\n", my_name.c_str());
        st = agent->sendLocalMD(&md_args);
        if (st != NIXL_SUCCESS) {
            fprintf(stderr, "[%s] sendLocalMD failed: %d\n", my_name.c_str(), st);
            return st;
        }
        fprintf(stderr, "[%s] metadata exchange complete\n", my_name.c_str());
    } else {
        fprintf(stderr, "[%s] listen thread will handle incoming metadata\n",
                my_name.c_str());
    }

    // 7. Address exchange via NIXL notifications.
    //
    //    We need each side's recv_buf device address and GPU id to build
    //    nixlRemoteDesc correctly.
    //    genNotif sends a small blob to the peer; getNotifs drains the local
    //    inbox.  genNotif returns non-SUCCESS until the peer's metadata is
    //    loaded locally, so spinning on it is the correct wait for the receiver
    //    (whose listen thread loads the sender's MD asynchronously).
    PeerRecvInfo my_peer_info{};
    my_peer_info.recv_addr = reinterpret_cast<uintptr_t>(recv_buf);
    my_peer_info.gpu_id = static_cast<uint64_t>(params.gpu_id);

    nixl_blob_t peer_info_blob(sizeof(PeerRecvInfo), '\0');
    memcpy(peer_info_blob.data(), &my_peer_info, sizeof(PeerRecvInfo));

    // Wait silently until the listen thread has loaded the peer's metadata.
    // checkRemoteMD returns NIXL_ERR_NOT_FOUND (without logging) until
    // remoteBackends_ is populated; genNotif would log ERROR on every retry.
    nixl_xfer_dlist_t empty_dlist(VRAM_SEG);
    fprintf(stderr, "[%s] waiting for peer metadata to be ready...\n", my_name.c_str());
    while (agent->checkRemoteMD(peer_name, empty_dlist) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    nixl_opt_args_t notif_args;
    notif_args.backends.push_back(ucx_backend);

    fprintf(stderr, "[%s] sending recv_buf addr to peer...\n", my_name.c_str());
    while ((st = agent->genNotif(peer_name, peer_info_blob, &notif_args)) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    fprintf(stderr, "[%s] waiting for peer recv_buf addr...\n", my_name.c_str());
    nixl_notifs_t notifs;
    bool logged_unexpected_notifs = false;
    const auto notif_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
        st = agent->getNotifs(notifs, &notif_args);
        if (st != NIXL_SUCCESS) {
            fprintf(stderr, "[%s] getNotifs failed while waiting for peer recv_buf addr: %d\n",
                    my_name.c_str(), st);
            return st;
        }

        const auto peer_notifs = notifs.find(peer_name);
        if (peer_notifs != notifs.end() && !peer_notifs->second.empty()) {
            break;
        }

        if (!notifs.empty() && !logged_unexpected_notifs) {
            fprintf(stderr, "[%s] received notifications, but none from expected peer '%s':",
                    my_name.c_str(), peer_name.c_str());
            for (const auto &entry : notifs) {
                fprintf(stderr, " '%s'(%zu)", entry.first.c_str(), entry.second.size());
            }
            fprintf(stderr, "\n");
            logged_unexpected_notifs = true;
        }

        if (std::chrono::steady_clock::now() >= notif_deadline) {
            fprintf(stderr, "[%s] timed out waiting for peer recv_buf addr from '%s'\n",
                    my_name.c_str(), peer_name.c_str());
            return NIXL_ERR_NOT_FOUND;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto &peer_notifs = notifs[peer_name];
    if (peer_notifs[0].size() < sizeof(PeerRecvInfo)) {
        fprintf(stderr, "[%s] peer info notification too small (%zu bytes)\n",
                my_name.c_str(), peer_notifs[0].size());
        return NIXL_ERR_MISMATCH;
    }

    PeerRecvInfo peer_info{};
    memcpy(&peer_info, peer_notifs[0].data(), sizeof(PeerRecvInfo));
    fprintf(stderr, "[%s] peer recv_buf addr: 0x%lx gpu=%lu\n",
            my_name.c_str(), peer_info.recv_addr, peer_info.gpu_id);

    // 8. Build memory view handles. PUT needs both a local source and remote
    //    destination view; atomic-flag only needs the remote counter view.
    nixl_remote_dlist_t remote_dlist(VRAM_SEG);
    remote_dlist.addDesc(
        nixlRemoteDesc(peer_info.recv_addr, buf_size, peer_info.gpu_id, peer_name));

    // Retry briefly while the remote metadata settles, but fail cleanly for
    // unsupported memory/backend combinations instead of spinning forever.
    if (params.op == gpu_bench_op::Put) {
        nixl_local_dlist_t local_send_dlist(VRAM_SEG);
        local_send_dlist.addDesc(nixlBasicDesc((uintptr_t)send_buf, buf_size, params.gpu_id));
        st = prep_memview_with_retries(agent.get(), local_send_dlist, local_mvh,
                                       my_name, "local");
        if (st != NIXL_SUCCESS) return st;
    }
    st = prep_memview_with_retries(agent.get(), remote_dlist, remote_mvh,
                                   my_name, "remote");
    if (st != NIXL_SUCCESS) return st;
    fprintf(stderr, "[%s] memory views ready — setup complete\n", my_name.c_str());

    return NIXL_SUCCESS;
}

// ---- BenchContext::~BenchContext ---------------------------------------------

BenchContext::~BenchContext()
{
    if (!agent) return;

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    // Clear the global __device__ pointer before the proxy runtime backing it
    // is destroyed by the agent's destructor.
    bench_proxy_clear_context();
#endif

    if (local_mvh)  agent->releaseMemView(local_mvh);
    if (remote_mvh) agent->releaseMemView(remote_mvh);

    if (send_buf && buf_size > 0) {
        nixl_reg_dlist_t send_dlist(VRAM_SEG);
        send_dlist.addDesc(nixlBlobDesc((uintptr_t)send_buf, buf_size, gpu_id, ""));
        agent->deregisterMem(send_dlist);
        cudaFree(send_buf);
    }
    if (recv_buf && buf_size > 0) {
        nixl_reg_dlist_t recv_dlist(VRAM_SEG);
        recv_dlist.addDesc(nixlBlobDesc((uintptr_t)recv_buf, buf_size, gpu_id, ""));
        agent->deregisterMem(recv_dlist);
        cudaFree(recv_buf);
    }

    // agent unique_ptr destroyed after this body — agent outlives the cleanup above.
}
