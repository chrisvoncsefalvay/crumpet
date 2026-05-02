// CRUMPET: shared CUDA helpers for fused 3D shifted-window kernels.
// SPDX-License-Identifier: MIT
//
// Contents:
//   - crumpet::FastDivmod
//       CUTLASS-style 32-bit unsigned reciprocal-multiply division. Replaces
//       runtime IDIV/IMOD by ws_d/ws_h/ws_w/windows_w/windows_h/volume in the
//       per-position address computation. On SM_80+ a single MUL.HI.U32 +
//       SHF retires in ~6 cycles versus 20-30+ cycles for IDIV through the
//       fp64 unit. Granlund & Montgomery, PLDI 1994.
//
//   - crumpet::ldg128 / crumpet::stg128
//       Inline-PTX 16-byte vector load (LDG.E.CI.128 via `ld.global.nc.v4.u32`)
//       and store (STG.E.128 via `st.global.v4.u32`). The non-coherent load
//       routes through the read-only L1 path, freeing the writeable L1 cache
//       for the output side of the same kernel and increasing achievable
//       MIO/L2 throughput on access-skewed workloads.
//
//   - crumpet::VecOps<scalar_t, kVecBytes>
//       Generic 16/8/4-byte vector load/store dispatch, used by the partition
//       kernels to vectorise the C-dim copy regardless of dtype.
//
// All identifiers live in the `crumpet` namespace to avoid spilling into the
// torch dispatcher's anonymous namespace.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace crumpet {

// ---------------------------------------------------------------------------
// FastDivmod: 32-bit unsigned reciprocal multiply.
//
// Constructed host-side; passed by value into the kernel as a small POD. The
// device-side `divmod` is two MULs and a SHF.
// ---------------------------------------------------------------------------
struct FastDivmod {
    int32_t  divisor;
    uint32_t multiplier;
    uint32_t shift_right;

    __host__ __device__ FastDivmod() : divisor(1), multiplier(1u), shift_right(0u) {}

    __host__ explicit FastDivmod(int32_t d) : divisor(d), multiplier(0u), shift_right(0u) {
        if (d == 1) {
            multiplier  = 1u;
            shift_right = 0u;
            return;
        }
        // Smallest l such that 2^l >= d  (= ceil(log2(d))).
        uint32_t l = 0;
        while ((1u << l) < static_cast<uint32_t>(d)) ++l;
        // m = ceil(2^(31+l) / d). Done in 64 bits to avoid the overflow that
        // bites the naive 32-bit Granlund derivation when d is a power of two.
        const uint64_t two_p = uint64_t(1) << (31 + l);
        const uint64_t m = (two_p + uint64_t(d) - 1) / uint64_t(d);
        multiplier  = static_cast<uint32_t>(m);
        shift_right = l - 1;
    }

    __host__ __device__ __forceinline__ int32_t div(int32_t x) const {
#if defined(__CUDA_ARCH__)
        if (divisor == 1) return x;
        const uint32_t q = __umulhi(static_cast<uint32_t>(x), multiplier);
        return static_cast<int32_t>(q >> shift_right);
#else
        return x / divisor;
#endif
    }

    __host__ __device__ __forceinline__ int32_t mod(int32_t x) const {
        return x - div(x) * divisor;
    }

    // Fused divmod. Returns quotient in `q` and remainder in `r`.
    __host__ __device__ __forceinline__ void divmod(int32_t& q, int32_t& r, int32_t x) const {
        q = div(x);
        r = x - q * divisor;
    }
};

// ---------------------------------------------------------------------------
// 128-bit aligned global memory ops via inline PTX.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 ldg128(const uint4* __restrict__ ptr) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ void stg128(uint4* __restrict__ ptr, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 ::
                 "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ uint2 ldg64(const uint2* __restrict__ ptr) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ void stg64(uint2* __restrict__ ptr, uint2 v) {
    asm volatile("st.global.v2.u32 [%0], {%1,%2};"
                 ::
                 "l"(ptr), "r"(v.x), "r"(v.y));
}

__device__ __forceinline__ uint32_t ldg32(const uint32_t* __restrict__ ptr) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ void stg32(uint32_t* __restrict__ ptr, uint32_t v) {
    asm volatile("st.global.u32 [%0], %1;" :: "l"(ptr), "r"(v));
}

// ---------------------------------------------------------------------------
// Vectorised load/store dispatch: exposes a `vec_t` register type and
// independent `load` / `store` operations so the kernel can split a copy
// into a load phase followed by a store phase. Doing so keeps multiple
// outstanding global memory transactions in flight per thread, which is
// what the scheduler needs to hide L1TEX long_scoreboard latency on GB10
// (Blackwell ld.global.nc latency is ~100+ cycles unless overlapped).
// ---------------------------------------------------------------------------
template <int kVecBytes>
struct VecOps;

template <>
struct VecOps<16> {
    using vec_t = uint4;
    template <typename scalar_t>
    __device__ __forceinline__ static vec_t load(const scalar_t* __restrict__ src) {
        return ldg128(reinterpret_cast<const uint4*>(src));
    }
    template <typename scalar_t>
    __device__ __forceinline__ static void store(scalar_t* __restrict__ dst, vec_t v) {
        stg128(reinterpret_cast<uint4*>(dst), v);
    }
};

template <>
struct VecOps<8> {
    using vec_t = uint2;
    template <typename scalar_t>
    __device__ __forceinline__ static vec_t load(const scalar_t* __restrict__ src) {
        return ldg64(reinterpret_cast<const uint2*>(src));
    }
    template <typename scalar_t>
    __device__ __forceinline__ static void store(scalar_t* __restrict__ dst, vec_t v) {
        stg64(reinterpret_cast<uint2*>(dst), v);
    }
};

template <>
struct VecOps<4> {
    using vec_t = uint32_t;
    template <typename scalar_t>
    __device__ __forceinline__ static vec_t load(const scalar_t* __restrict__ src) {
        return ldg32(reinterpret_cast<const uint32_t*>(src));
    }
    template <typename scalar_t>
    __device__ __forceinline__ static void store(scalar_t* __restrict__ dst, vec_t v) {
        stg32(reinterpret_cast<uint32_t*>(dst), v);
    }
};

template <>
struct VecOps<2> {
    // 2-byte fallback (e.g. fp16/bf16 with C = 1). Used only for pathological
    // shapes — never on the BTCV hot path. We skip the read-only path here
    // because LDG.E.NC.U16 isn't a 1:1 PTX intrinsic; a normal scalar copy is
    // both correct and good enough for the fallback.
    using vec_t = uint16_t;
    template <typename scalar_t>
    __device__ __forceinline__ static vec_t load(const scalar_t* __restrict__ src) {
        return *reinterpret_cast<const uint16_t*>(src);
    }
    template <typename scalar_t>
    __device__ __forceinline__ static void store(scalar_t* __restrict__ dst, vec_t v) {
        *reinterpret_cast<uint16_t*>(dst) = v;
    }
};

// Fused load + store helper, kept for any caller that doesn't need the
// load / store split (e.g. small-C tail loops).
template <int kVecBytes>
struct VecCopy {
    template <typename scalar_t>
    __device__ __forceinline__ static void copy(
        scalar_t* __restrict__ dst, const scalar_t* __restrict__ src) {
        const auto v = VecOps<kVecBytes>::template load<scalar_t>(src);
        VecOps<kVecBytes>::template store<scalar_t>(dst, v);
    }
};

// Selects the widest aligned vector width given `bytes` (= C * sizeof(scalar)).
__host__ __forceinline__ int select_vec_bytes(int bytes) {
    if ((bytes & 0xF) == 0) return 16;
    if ((bytes & 0x7) == 0) return 8;
    if ((bytes & 0x3) == 0) return 4;
    return static_cast<int>(sizeof(uint16_t));  // 2-byte fallback
}

}  // namespace crumpet
