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
#include "ucx_proxy_backend.h"

// Shape-only handoff: keep the adapter lifecycle surface without preserving
// the UCX proxy backend behavior in this scaffold. The adapter still receives
// resolved backend memviews after proxy-ID resolution in the common runtime.

nixl_status_t
nixlUcxProxyBackend::init(uint32_t worker_count, uint32_t channel_count) {
    worker_count_ = worker_count;
    channel_count_ = channel_count;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::loadRemoteConnInfo(const std::string &remote_name,
                                        const nixl_blob_t &conn_info) {
    (void)remote_name;
    (void)conn_info;
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlUcxProxyBackend::submit(const ResolvedProxySubmission &, uint64_t &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlUcxProxyBackend::checkCompletion(uint64_t) {
    return NIXL_IN_PROG;
}

size_t
nixlUcxProxyBackend::progress() {
    return 0;
}

nixl_status_t
nixlUcxProxyBackend::shutdown() {
    return NIXL_SUCCESS;
}
