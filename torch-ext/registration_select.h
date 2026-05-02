// CRUMPET
// SPDX-License-Identifier: MIT
// Prefer kernel-builder's generated registration.h when it is available.
#pragma once

#if __has_include(<registration.h>)
#include <registration.h>
#else
#include "local_registration.h"
#endif

