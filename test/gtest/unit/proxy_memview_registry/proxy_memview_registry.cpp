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
#include <cstdint>
#include <thread>
#include <vector>

#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_memview_registry {

    class ProxyMemViewRegistryTest : public testing::Test {
    protected:
        ProxyMemViewRegistry registry_;

        nixlMemViewH
        makeFakeBackendHandle(uint64_t id) {
            return reinterpret_cast<nixlMemViewH>(id);
        }
    };

    TEST_F(ProxyMemViewRegistryTest, RegisterSingle) {
        nixlMemViewH proxy_handle = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(100), &proxy_handle),
                  NIXL_SUCCESS);
        EXPECT_NE(proxy_handle, nullptr);
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterNullOutputReturnsError) {
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(100), nullptr),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterMultipleAssignsUniqueIds) {
        nixlMemViewH h1 = nullptr, h2 = nullptr, h3 = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &h1), NIXL_SUCCESS);
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &h2), NIXL_SUCCESS);
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(30), &h3), NIXL_SUCCESS);

        EXPECT_NE(h1, h2);
        EXPECT_NE(h2, h3);
        EXPECT_NE(h1, h3);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveByHandle) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(proxy_handle, resolved));
        EXPECT_EQ(resolved, backend);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveById) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        auto proxy_id = reinterpret_cast<uint64_t>(proxy_handle);
        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemViewId(proxy_id, resolved));
        EXPECT_EQ(resolved, backend);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveMultiple) {
        auto b1 = makeFakeBackendHandle(10), b2 = makeFakeBackendHandle(20);
        nixlMemViewH h1 = nullptr, h2 = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(b1, &h1), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(b2, &h2), NIXL_SUCCESS);

        nixlMemViewH r1 = nullptr, r2 = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(h1, r1));
        EXPECT_TRUE(registry_.resolveProxyMemView(h2, r2));
        EXPECT_EQ(r1, b1);
        EXPECT_EQ(r2, b2);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveUnregisteredHandleReturnsFalse) {
        auto bogus = makeFakeBackendHandle(999);
        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemView(bogus, resolved));
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveNullHandleReturnsFalse) {
        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemView(nullptr, resolved));
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveIdZeroReturnsFalse) {
        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemViewId(0, resolved));
    }

    TEST_F(ProxyMemViewRegistryTest, UnregisterThenResolveFails) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        EXPECT_EQ(registry_.unregisterProxyMemView(proxy_handle), NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemView(proxy_handle, resolved));
    }

    TEST_F(ProxyMemViewRegistryTest, UnregisterDoesNotAffectOtherEntries) {
        auto b1 = makeFakeBackendHandle(10), b2 = makeFakeBackendHandle(20);
        nixlMemViewH h1 = nullptr, h2 = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(b1, &h1), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(b2, &h2), NIXL_SUCCESS);

        EXPECT_EQ(registry_.unregisterProxyMemView(h1), NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemView(h1, resolved));
        EXPECT_TRUE(registry_.resolveProxyMemView(h2, resolved));
        EXPECT_EQ(resolved, b2);
    }

    TEST_F(ProxyMemViewRegistryTest, UnregisterNullHandleReturnsError) {
        EXPECT_EQ(registry_.unregisterProxyMemView(nullptr), NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, UnregisterInvalidHandleReturnsError) {
        auto bogus = makeFakeBackendHandle(999);
        EXPECT_EQ(registry_.unregisterProxyMemView(bogus), NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, ClearResetsAllEntries) {
        nixlMemViewH h1 = nullptr, h2 = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &h1), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &h2), NIXL_SUCCESS);

        registry_.clear();

        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(registry_.resolveProxyMemView(h1, resolved));
        EXPECT_FALSE(registry_.resolveProxyMemView(h2, resolved));
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterAfterClearWorks) {
        nixlMemViewH h1 = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &h1), NIXL_SUCCESS);

        registry_.clear();

        nixlMemViewH h2 = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &h2), NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(h2, resolved));
        EXPECT_EQ(resolved, makeFakeBackendHandle(20));
    }

    TEST_F(ProxyMemViewRegistryTest, ConcurrentRegistrations) {
        constexpr int kThreads = 8;
        constexpr int kPerThread = 100;

        std::vector<std::thread> threads;
        std::vector<std::vector<nixlMemViewH>> thread_handles(kThreads);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([this, t, &thread_handles]() {
                for (int i = 0; i < kPerThread; ++i) {
                    nixlMemViewH proxy_handle = nullptr;
                    auto backend = makeFakeBackendHandle(t * kPerThread + i + 1);
                    EXPECT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);
                    thread_handles[t].push_back(proxy_handle);
                }
            });
        }

        for (auto &th : threads) {
            th.join();
        }

        for (int t = 0; t < kThreads; ++t) {
            for (int i = 0; i < kPerThread; ++i) {
                auto expected_backend = makeFakeBackendHandle(t * kPerThread + i + 1);
                nixlMemViewH resolved = nullptr;
                EXPECT_TRUE(registry_.resolveProxyMemView(thread_handles[t][i], resolved));
                EXPECT_EQ(resolved, expected_backend);
            }
        }
    }

} // namespace proxy_memview_registry
} // namespace gtest
