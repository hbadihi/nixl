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
#include <gtest/gtest.h>

#include "nixl.h"

namespace {

/**
 * Engine-owned proxy lifecycle: the runtime is created and torn down purely
 * via UCX backend params, with no proxy configuration on the agent.
 */
TEST(ProxyBackendParamsTest, CreateAndTeardown) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_create", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"}};
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
    ASSERT_NE(backend, nullptr);
    // Teardown (agent destructor) must join the proxy threads cleanly.
}

TEST(ProxyBackendParamsTest, MatchingNumWorkersAccepted) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_match", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"},
                           {"num_workers", "8"}};
    nixlBackendH *backend = nullptr;
    EXPECT_EQ(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

TEST(ProxyBackendParamsTest, NumWorkersMismatchRejected) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_mismatch", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"},
                           {"num_workers", "3"}};
    nixlBackendH *backend = nullptr;
    EXPECT_NE(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

TEST(ProxyBackendParamsTest, UnknownProxyParamRejected) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_unknown", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_worker_count", "2"}};
    nixlBackendH *backend = nullptr;
    EXPECT_NE(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

} // namespace
