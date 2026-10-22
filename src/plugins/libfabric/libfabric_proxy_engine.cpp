/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "libfabric_proxy_engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/nixl_log.h"
#include "device/device_memview.h"
#include "device/device_ops.h"
#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_runtime.h"
#include "device/proxy/proxy_transport.h"

namespace nixl::libfabric {
class proxyEngine::transport final : public nixl::proxyTransport {
public:
    transport(nixlLibfabricRailManager &rails, const nixl::proxyConfig &config)
        : rails_(rails),
          max_peers_(config.max_peers),
          thread_count_(config.thread_count),
          threads_(config.thread_count),
          rings_(config.ringCount()) {
        for (threadState &thread : threads_) {
            thread.in_flight.resize(rails_.getNumRails());
        }
        for (size_t rail_id = 0; rail_id < rails_.getNumRails(); ++rail_id) {
            threads_[rail_id % thread_count_].rails.push_back(rail_id);
        }
        for (uint32_t channel = 0; channel < config.channel_count; ++channel) {
            for (uint32_t peer = 0; peer < max_peers_; ++peer) {
                ringState &ring = rings_[ringIndex(channel, peer)];
                ring.thread = config.owningThread(channel, peer);
                const nixl::proxyThreadShare share = config.threadShare(ring.thread);
                ring.anchor = channel == share.first_channel && peer == share.peers.first;
                ring.ops.resize(config.ring_depth);
                for (op &record : ring.ops) {
                    record.ring = &ring;
                }
            }
        }
    }

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &submission,
           nixl::proxyBackendRequest &request) override {
        request = nixl::proxyBackendRequest{};
        ringState &ring = rings_[ringIndex(submission.channel_id, submission.peer_index)];
        op *record = nullptr;
        nixl_status_t status = NIXL_ERR_NOT_SUPPORTED;
        switch (submission.opcode) {
        case nixl_proxy_opcode_t::PUT:
            status = postPut(submission, ring, record);
            break;
        case nixl_proxy_opcode_t::ATOMIC_ADD:
            status = postAtomicAdd(submission, ring, record);
            break;
        }
        if (status == NIXL_IN_PROG) {
            request.token = reinterpret_cast<uintptr_t>(record);
        }
        return status;
    }

    nixl_status_t
    checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &request) override {
        if (!request) {
            return NIXL_ERR_INVALID_PARAM;
        }
        op *record = reinterpret_cast<op *>(request.token);
        if (!record->done) {
            return NIXL_IN_PROG;
        }
        record->busy = false;
        return record->status;
    }

    void
    progress(uint32_t channel, uint32_t peer) noexcept override {
        const ringState &ring = rings_[ringIndex(channel, peer)];
        if (ring.anchor) {
            threadState &thread = threads_[ring.thread];
            pollRails(thread, ++thread.passes % kSweepPasses == 0);
        }
    }

    nixl_status_t
    quiesce(uint32_t channel, uint32_t peer) override {
        const ringState &ring = rings_[ringIndex(channel, peer)];
        while (ring.in_flight != 0) {
            pollRails(threads_[ring.thread], false);
        }
        return NIXL_SUCCESS;
    }

private:
    struct ringState;

    static constexpr uint32_t kSweepPasses = 64;

    struct op {
        ringState *ring = nullptr;
        uint32_t *rail_in_flight = nullptr;
        uint64_t operand = 0;
        nixl_status_t status = NIXL_IN_PROG;
        bool busy = false;
        bool done = false;

        void
        complete(nixl_status_t result) noexcept {
            status = result;
            done = true;
            --ring->in_flight;
            --*rail_in_flight;
        }
    };

    struct alignas(64) ringState {
        std::vector<op> ops;
        size_t next_op = 0;
        size_t next_rail = 0;
        size_t next_endpoint = 0;
        uint32_t in_flight = 0;
        uint32_t thread = 0;
        bool anchor = false;
    };

    struct alignas(64) threadState {
        std::vector<size_t> rails;
        std::vector<uint32_t> in_flight;
        uint32_t passes = 0;
    };

    [[nodiscard]] size_t
    ringIndex(uint32_t channel, uint32_t peer) const noexcept {
        return static_cast<size_t>(channel) * max_peers_ + peer;
    }

    void
    pollRails(const threadState &thread, bool sweep) noexcept {
        nixlLibfabricRail::setProgressingThread(true);
        for (const size_t rail_id : thread.rails) {
            if (sweep || thread.in_flight[rail_id] != 0) {
                rails_.getRail(rail_id).progressCompletionQueue();
            }
        }
    }

    [[nodiscard]] op *
    allocateOp(ringState &ring) noexcept {
        op &record = ring.ops[ring.next_op];
        if (record.busy) {
            NIXL_ERROR << "device proxy: more requests in flight than the ring depth";
            return nullptr;
        }
        ring.next_op = (ring.next_op + 1) % ring.ops.size();
        record.busy = true;
        record.done = false;
        record.status = NIXL_IN_PROG;
        return &record;
    }

    [[nodiscard]] int
    selectRail(ringState &ring, const std::vector<size_t> &selected_rails) const noexcept {
        size_t owned = 0;
        for (const size_t rail_id : selected_rails) {
            owned += rail_id % thread_count_ == ring.thread;
        }
        if (owned == 0) {
            return -1;
        }
        size_t pick = ring.next_rail++ % owned;
        for (const size_t rail_id : selected_rails) {
            if (rail_id % thread_count_ == ring.thread && pick-- == 0) {
                return static_cast<int>(rail_id);
            }
        }
        return -1;
    }

    nixl_status_t
    postPut(const nixl::proxyBackendSubmission &submission, ringState &ring, op *&record) {
        const nixlMetaDesc &local = submission.local;
        const nixlMetaDesc &remote = submission.remote;
        if (local.len != submission.size || remote.len != submission.size) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const auto *lmd = static_cast<const nixlLibfabricPrivateMetadata *>(local.metadataP);
        const auto *rmd = static_cast<const nixlLibfabricPublicMetadata *>(remote.metadataP);
        if (lmd == nullptr || rmd == nullptr || rmd->conn_ == nullptr ||
            rmd->remote_selected_endpoints_.empty()) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const int rail_id = selectRail(ring, lmd->selected_rails_);
        if (rail_id < 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto &endpoints = rmd->remote_selected_endpoints_;
        const size_t remote_ep = endpoints[ring.next_endpoint++ % endpoints.size()];
        nixlLibfabricRail &rail = rails_.getRail(rail_id);
        const fi_addr_t dest = rmd->conn_->rail_remote_addr_list_.at(rail_id)[remote_ep];
        uint64_t remote_addr = remote.addr;
        if (!(rail.getRailInfo()->domain_attr->mr_mode & FI_MR_VIRT_ADDR)) {
            remote_addr -= rmd->remote_buf_addr_;
        }

        record = allocateOp(ring);
        if (record == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 0);
        if (req == nullptr) {
            record->busy = false;
            return NIXL_ERR_BACKEND;
        }
        req->completion_callback = [record](nixl_status_t status) { record->complete(status); };

        nixl_status_t status;
        while ((status = rail.tryPostWrite(reinterpret_cast<void *>(local.addr),
                                           submission.size,
                                           fi_mr_desc(lmd->rail_mr_list_[rail_id]),
                                           dest,
                                           remote_addr,
                                           rmd->rail_remote_key_list_[remote_ep],
                                           req,
                                           FI_DELIVERY_COMPLETE)) == NIXL_IN_PROG) {
            rail.progressCompletionQueue();
        }
        return finishPost(rail, req, ring, record, status);
    }

    nixl_status_t
    postAtomicAdd(const nixl::proxyBackendSubmission &submission, ringState &ring, op *&record) {
        const nixlMetaDesc &remote = submission.remote;
        if (remote.len != sizeof(uint64_t)) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const auto *rmd = static_cast<const nixlLibfabricPublicMetadata *>(remote.metadataP);
        if (rmd == nullptr || rmd->conn_ == nullptr || rmd->remote_selected_endpoints_.empty()) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const size_t rail_id = threads_[ring.thread].rails.front();
        nixlLibfabricRail &rail = rails_.getRail(rail_id);
        if (!(rail.getRailInfo()->caps & FI_ATOMIC)) {
            NIXL_ERROR << "device proxy: rail " << rail_id << " provider has no FI_ATOMIC";
            return NIXL_ERR_NOT_SUPPORTED;
        }
        const auto &endpoints = rmd->remote_selected_endpoints_;
        const size_t remote_ep = endpoints[(remote.addr / sizeof(uint64_t)) % endpoints.size()];
        const fi_addr_t dest = rmd->conn_->rail_remote_addr_list_.at(rail_id)[remote_ep];
        uint64_t remote_addr = remote.addr;
        if (!(rail.getRailInfo()->domain_attr->mr_mode & FI_MR_VIRT_ADDR)) {
            remote_addr -= rmd->remote_buf_addr_;
        }

        while (ring.in_flight != 0) {
            pollRails(threads_[ring.thread], false);
        }

        record = allocateOp(ring);
        if (record == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        record->operand = submission.value;
        nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 0);
        if (req == nullptr) {
            record->busy = false;
            return NIXL_ERR_BACKEND;
        }
        req->completion_callback = [record](nixl_status_t status) { record->complete(status); };

        nixl_status_t status;
        while ((status = rail.tryPostAtomicAdd(&record->operand,
                                               dest,
                                               remote_addr,
                                               rmd->rail_remote_key_list_[remote_ep],
                                               req,
                                               FI_DELIVERY_COMPLETE)) == NIXL_IN_PROG) {
            rail.progressCompletionQueue();
        }
        return finishPost(rail, req, ring, record, status);
    }

    nixl_status_t
    finishPost(nixlLibfabricRail &rail,
               nixlLibfabricReq *req,
               ringState &ring,
               op *record,
               nixl_status_t status) {
        if (status != NIXL_SUCCESS) {
            rail.releaseRequest(req);
            record->busy = false;
            return status;
        }
        record->rail_in_flight = &threads_[ring.thread].in_flight[rail.rail_id];
        ++*record->rail_in_flight;
        ++ring.in_flight;
        return NIXL_IN_PROG;
    }

    nixlLibfabricRailManager &rails_;
    const uint32_t max_peers_;
    const uint32_t thread_count_;
    std::vector<threadState> threads_;
    std::vector<ringState> rings_;
};
} // namespace nixl::libfabric

namespace {
[[nodiscard]] nixl::deviceOps &
requireDeviceOps() {
    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        nixl::throwRuntimeError(
            "device proxy needs a device operations implementation; none is loaded");
    }
    return *ops;
}

constexpr uint32_t kRingSubmitsPerPass = 16;

[[nodiscard]] nixl::proxyConfig
runtimeConfig(const nixl::proxyConfig &config) noexcept {
    nixl::proxyConfig runtime_config = config;
    runtime_config.ring_submits_per_pass = kRingSubmitsPerPass;
    return runtime_config;
}
} // namespace

namespace nixl::libfabric {
proxyEngine::proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config)
    : nixlLibfabricEngine(&init_params),
      ops_(requireDeviceOps()),
      thread_count_(config.thread_count) {
    const size_t num_rails = railManager().getNumRails();
    if (thread_count_ > num_rails) {
        nixl::throwRuntimeError("proxy_thread_count=",
                                thread_count_,
                                " exceeds the ",
                                num_rails,
                                " libfabric rails; each proxy thread owns at least one rail");
    }

    bool emulated_write = false;
    size_t optlen = sizeof(emulated_write);
    if (fi_getopt(&railManager().getRail(0).endpoint->fid,
                  FI_OPT_ENDPOINT,
                  FI_OPT_EFA_EMULATED_WRITE,
                  &emulated_write,
                  &optlen) == 0 &&
        emulated_write) {
        NIXL_WARN << "device proxy: the provider emulates RDMA writes, so the target CPU "
                  << "handles every put";
    }

    setExternalProgress(true);
    runtime_ = nixl::proxyRuntime::create(
        std::make_unique<transport>(railManager(), config), runtimeConfig(config), ops_);
    NIXL_INFO << "Engine-owned device proxy enabled: " << config.channel_count << " channel(s), "
              << config.thread_count << " thread(s) over " << num_rails
              << " rail(s), max_peers=" << config.max_peers << ", ring_depth=" << config.ring_depth;
}

proxyEngine::~proxyEngine() {
    runtime_.reset();
    setExternalProgress(false);
}

nixl_status_t
proxyEngine::disconnect(const std::string &remote_agent) {
    runtime_->drain();
    return nixlLibfabricEngine::disconnect(remote_agent);
}

nixl_status_t
proxyEngine::prepXfer(const nixl_xfer_op_t &,
                      const nixl_meta_dlist_t &,
                      const nixl_meta_dlist_t &,
                      const std::string &,
                      nixlBackendReqH *&,
                      const nixl_opt_b_args_t *) const {
    NIXL_ERROR << "host transfers are not supported with device_proxy=true";
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
proxyEngine::postXfer(const nixl_xfer_op_t &,
                      const nixl_meta_dlist_t &,
                      const nixl_meta_dlist_t &,
                      const std::string &,
                      nixlBackendReqH *&,
                      const nixl_opt_b_args_t *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
proxyEngine::checkXfer(nixlBackendReqH *) const {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<typename DlistT>
nixl_status_t
proxyEngine::prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const {
    nixlMemViewH proxy_mvh = nullptr;
    nixl_status_t status = runtime_->prepMemView(dlist, &proxy_mvh);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = nixl::deviceMemViewAllocate(ops_, nixl_device_exec_mode_t::PROXY, proxy_mvh, mvh);
    if (status != NIXL_SUCCESS) {
        runtime_->releaseMemView(proxy_mvh);
    }
    return status;
}

nixl_status_t
proxyEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    return prepProxyMemView(dlist, mvh);
}

nixl_status_t
proxyEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    for (const auto &desc : dlist) {
        const auto *md = static_cast<const nixlLibfabricPrivateMetadata *>(desc.metadataP);
        if (md == nullptr) {
            continue;
        }
        for (uint32_t thread = 0; thread < thread_count_; ++thread) {
            bool owned = false;
            for (const size_t rail_id : md->selected_rails_) {
                owned = owned || rail_id % thread_count_ == thread;
            }
            if (!owned) {
                NIXL_ERROR << "device proxy thread " << thread << " owns none of the rails of "
                           << "buffer " << reinterpret_cast<void *>(desc.addr)
                           << "; lower proxy_thread_count to at most the rails near its device";
                return NIXL_ERR_INVALID_PARAM;
            }
        }
    }
    return prepProxyMemView(dlist, mvh);
}

void
proxyEngine::releaseMemView(nixlMemViewH mem_view) const {
    if (mem_view == nullptr) {
        return;
    }

    nixlMemViewH proxy_mvh = nullptr;
    if (nixl::deviceMemViewGetBackend(ops_, mem_view, proxy_mvh) != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to read proxy memory view before retirement";
    }

    runtime_->releaseMemView(proxy_mvh);
    nixl::deviceMemViewFree(ops_, mem_view);
}
} // namespace nixl::libfabric
