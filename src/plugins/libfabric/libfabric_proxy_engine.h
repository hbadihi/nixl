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

#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_ENGINE_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_PROXY_ENGINE_H

#include <cstdint>
#include <memory>
#include <string>

#include "libfabric_backend.h"

namespace nixl {
class deviceOps;
class proxyRuntime;
struct proxyConfig;
} // namespace nixl

namespace nixl::libfabric {
/**
 * The libfabric engine with the device proxy (device_proxy=true): memory views run through the
 * proxy runtime. Rail r belongs to proxy thread r % proxy_thread_count, which alone posts proxy
 * operations on it and polls it. Host transfers are not supported.
 */
class proxyEngine final : public nixlLibfabricEngine {
public:
    proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config);

    ~proxyEngine() override;

    /** Drains the proxy first: views naming the agent must be released before. */
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &dlist,
                nixlMemViewH &mvh,
                const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /** Fails unless every proxy thread owns a rail of each buffer. */
    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &dlist,
                nixlMemViewH &mvh,
                const nixl_opt_b_args_t *opt_args = nullptr) const override;

    void
    releaseMemView(nixlMemViewH mem_view) const override;

    /** Also maps VRAM through GDRCopy, so the proxy threads can apply atomic adds to it. */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

private:
    class transport;
    class atomicTargets;

    template<typename DlistT>
    nixl_status_t
    prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const;

    nixl::deviceOps &ops_;
    const uint32_t thread_count_;
    std::unique_ptr<atomicTargets> atomic_targets_;
    std::unique_ptr<nixl::proxyRuntime> runtime_;
};
} // namespace nixl::libfabric

#endif
