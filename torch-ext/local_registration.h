// CRUMPET
// SPDX-License-Identifier: MIT
// Local stub for kernel-builder's registration.h.
#pragma once

#ifndef CUDA_KERNEL
#define CUDA_KERNEL
#endif

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(name, m) TORCH_LIBRARY(name, m)
#endif

#define REGISTER_EXTENSION(name) /* noop for local JIT builds */

