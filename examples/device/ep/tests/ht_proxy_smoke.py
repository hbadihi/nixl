# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated HT proxy smoke entry point.

This wraps test_ht.py with smoke-sized defaults and emits evidence as the
`ht_proxy_smoke` validation path when --evidence-output is provided.
"""

from test_ht import main

if __name__ == "__main__":
    main(default_proxy_smoke=True)
