// CRUMPET: fused 3D shifted-window partition kernel.
// SPDX-License-Identifier: MIT
//
// Reads a `[B, D, H, W, C]` tensor and writes the cyclic-shifted, window-
// partitioned form `[B * num_windows, ws_d * ws_h * ws_w, C]`.
//
// Design notes (NCU-tuned on GB10 / SM_121 for the BTCV hot path):
//
// * 16-byte vectorised C-dim copy via inline-PTX `ld.global.nc.v4.u32` /
//   `st.global.v4.u32` (LDG.E.CI.128 / STG.E.128). The non-coherent load
//   routes through the read-only L1 path, freeing the writeable L1 cache
//   for the output side.
//
// * Sub-warp packing. The block layout is `(kSubWarpSize, kPosPerBlock)`
//   with `kPosPerBlock = kBlockSize / kSubWarpSize`. Each sub-warp owns
//   one output position; consecutive sub-warps within a CUDA warp
//   broadcast their lane-0 address through `__shfl_sync` instead of a
//   block-wide `__syncthreads`. `kSubWarpSize` is chosen at launch as
//   the smallest power of two `>= C / kVecElems`, capped at the warp
//   width — this keeps lane utilisation high while preserving sub-warp
//   coalescing on both the input gather and the output scatter.
//
// * CUTLASS-style FastDivmod replaces every runtime IDIV / IMOD in the
//   per-position address compute (`ws_d`, `ws_h`, `ws_w`, `windows_w`,
//   `windows_h`, `volume`). Each divmod is one MUL.HI.U32 plus a SHF.
//
// * `__launch_bounds__(kBlockSize, kBlocksPerSm)` caps the register
//   footprint so kBlocksPerSm blocks reside per SM at full occupancy.

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include "torch_binding.h"
#include "crumpet_common.cuh"

namespace {

using crumpet::FastDivmod;
using crumpet::VecCopy;
using crumpet::VecOps;

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 128;          // 4 warps per CTA
constexpr int kBlocksPerSm = 8;          // __launch_bounds__ occupancy target
constexpr unsigned kFullMask = 0xFFFFFFFFu;
// Per-thread iteration count over C_vec. The launcher chooses
// kSubWarpSize so that `ceil(C_vec / kSubWarpSize) <= kMaxILP`; on GB10
// the bandwidth-bound BTCV partition runs best with kMaxILP = 1 because
// adding ILP > 1 degrades coalescing more than it adds latency hiding.
constexpr int kMaxILP = 1;

// One sub-warp = one output position. `kSubWarpSize` is a compile-time
// power of two in {1, 2, 4, 8, 16, 32}. The block is `kBlockSize` threads
// wide; `kPosPerBlock = kBlockSize / kSubWarpSize` positions are packed
// into the y axis. The natural CUDA warp covers `(32 / kSubWarpSize)`
// consecutive rows so the broadcast shuffle stays inside a single warp.
template <typename scalar_t, bool Shifted, int kVecBytes, int kSubWarpSize>
__global__ void __launch_bounds__(kBlockSize, kBlocksPerSm)
shift_partition_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ x,
    const FastDivmod fdm_volume,
    const FastDivmod fdm_num_windows,
    const FastDivmod fdm_ws_w,
    const FastDivmod fdm_ws_h,
    const FastDivmod fdm_windows_w,
    const FastDivmod fdm_windows_h,
    const int positions,
    const int D, const int H, const int W, const int C,
    const int ws_d, const int ws_h, const int ws_w,
    const int ss_d, const int ss_h, const int ss_w) {

    static_assert(kSubWarpSize == 1 || kSubWarpSize == 2 || kSubWarpSize == 4 ||
                  kSubWarpSize == 8 || kSubWarpSize == 16 || kSubWarpSize == 32,
                  "kSubWarpSize must be a power of two and divide warp size");
    static_assert(kBlockSize % kSubWarpSize == 0,
                  "block size must be a whole multiple of kSubWarpSize");
    constexpr int kVecElems = (kVecBytes >= static_cast<int>(sizeof(scalar_t)))
        ? (kVecBytes / static_cast<int>(sizeof(scalar_t)))
        : 1;
    constexpr int kPosPerBlock = kBlockSize / kSubWarpSize;
    constexpr int kSubWarpsPerWarp = kWarpSize / kSubWarpSize;

    const int sub_lane = threadIdx.x;     // [0, kSubWarpSize)
    const int row = threadIdx.y;          // [0, kPosPerBlock)
    const int pos = blockIdx.x * kPosPerBlock + row;
    const bool active = (pos < positions);

    int src_base = 0;
    int out_base = 0;
    if (active && sub_lane == 0) {
        int tmp1, local;
        fdm_volume.divmod(tmp1, local, pos);
        int b, win_flat;
        fdm_num_windows.divmod(b, win_flat, tmp1);

        int loc_tmp, loc_w;
        fdm_ws_w.divmod(loc_tmp, loc_w, local);
        int loc_d, loc_h;
        fdm_ws_h.divmod(loc_d, loc_h, loc_tmp);

        int win_tmp, win_w;
        fdm_windows_w.divmod(win_tmp, win_w, win_flat);
        int win_d, win_h;
        fdm_windows_h.divmod(win_d, win_h, win_tmp);

        int src_d = win_d * ws_d + loc_d;
        int src_h = win_h * ws_h + loc_h;
        int src_w = win_w * ws_w + loc_w;
        if constexpr (Shifted) {
            src_d += ss_d;
            src_h += ss_h;
            src_w += ss_w;
            src_d -= (src_d >= D) ? D : 0;
            src_h -= (src_h >= H) ? H : 0;
            src_w -= (src_w >= W) ? W : 0;
        }
        src_base = (((b * D + src_d) * H + src_h) * W + src_w) * C;
        out_base = pos * C;
    }

    // Sub-warp broadcast inside the natural warp. Linear thread id within the
    // block puts kSubWarpsPerWarp consecutive rows into the same warp; the
    // leader of the sub-warp lives at lane (lane_in_warp / kSubWarpSize) *
    // kSubWarpSize. We use the full warp mask so the compiler keeps the
    // shuffle in flight regardless of the active predicate.
    const int lane_in_warp = (row * kSubWarpSize + sub_lane) & (kWarpSize - 1);
    const int leader_lane = (lane_in_warp / kSubWarpSize) * kSubWarpSize;
    src_base = __shfl_sync(kFullMask, src_base, leader_lane);
    out_base = __shfl_sync(kFullMask, out_base, leader_lane);

    if (!active) return;

    // C-dim copy. The launcher chooses `kSubWarpSize` so that the per-
    // thread iteration count is in [1, kMaxILP]. With kMaxILP = 1 each
    // thread issues one ld + one st of `kVecBytes`; with kMaxILP > 1 the
    // load phase is hoisted ahead of the store phase via a register
    // buffer, but on this workload the extra ILP costs more in coalescing
    // than it gains in latency hiding.
    const int C_vec = C / kVecElems;
    const scalar_t* x_pos = x + src_base;
    scalar_t* out_pos = output + out_base;

    using Vec = typename VecOps<kVecBytes>::vec_t;
    Vec buf[kMaxILP];

    #pragma unroll
    for (int u = 0; u < kMaxILP; ++u) {
        const int v = sub_lane + u * kSubWarpSize;
        if (v < C_vec) {
            buf[u] = VecOps<kVecBytes>::template load<scalar_t>(
                x_pos + v * kVecElems);
        }
    }
    #pragma unroll
    for (int u = 0; u < kMaxILP; ++u) {
        const int v = sub_lane + u * kSubWarpSize;
        if (v < C_vec) {
            VecOps<kVecBytes>::template store<scalar_t>(
                out_pos + v * kVecElems, buf[u]);
        }
    }

    // Spill loop for `kMaxILP * kSubWarpSize < C_vec`. The launcher
    // avoids this on the BTCV hot path; it exists for very wide C.
    #pragma unroll 1
    for (int v = sub_lane + kMaxILP * kSubWarpSize; v < C_vec; v += kSubWarpSize) {
        VecCopy<kVecBytes>::template copy<scalar_t>(
            out_pos + v * kVecElems, x_pos + v * kVecElems);
    }

    // Scalar tail when C is not a multiple of kVecElems.
    const int tail_start = C_vec * kVecElems;
    #pragma unroll 1
    for (int c = tail_start + sub_lane; c < C; c += kSubWarpSize) {
        out_pos[c] = x_pos[c];
    }
}

template <typename scalar_t, bool Shifted, int kVecBytes, int kSubWarpSize>
inline void launch_kernel(
    torch::Tensor& output,
    const torch::Tensor& x,
    const FastDivmod& fdm_volume,
    const FastDivmod& fdm_num_windows,
    const FastDivmod& fdm_ws_w,
    const FastDivmod& fdm_ws_h,
    const FastDivmod& fdm_windows_w,
    const FastDivmod& fdm_windows_h,
    int positions,
    int D, int H, int W, int C,
    int ws_d, int ws_h, int ws_w,
    int ss_d, int ss_h, int ss_w) {
    constexpr int kPosPerBlock = kBlockSize / kSubWarpSize;
    const dim3 threads(kSubWarpSize, kPosPerBlock);
    const int blocks = (positions + kPosPerBlock - 1) / kPosPerBlock;
    shift_partition_kernel<scalar_t, Shifted, kVecBytes, kSubWarpSize><<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        fdm_volume, fdm_num_windows,
        fdm_ws_w, fdm_ws_h,
        fdm_windows_w, fdm_windows_h,
        positions,
        D, H, W, C,
        ws_d, ws_h, ws_w,
        ss_d, ss_h, ss_w);
}

// Smallest power of two such that `ceil(C_vec / sub_warp) <= kMaxILP`,
// capped at the warp size. Drives both the per-warp position packing
// (`32 / sub_warp` positions) and the per-thread ILP (`C_vec / sub_warp`).
__host__ inline int select_subwarp(int c_vec) {
    const int target = (c_vec + kMaxILP - 1) / kMaxILP;
    if (target <= 1) return 1;
    if (target <= 2) return 2;
    if (target <= 4) return 4;
    if (target <= 8) return 8;
    if (target <= 16) return 16;
    return 32;
}

template <typename scalar_t, bool Shifted, int kVecBytes>
inline void dispatch_subwarp(
    torch::Tensor& output,
    const torch::Tensor& x,
    const FastDivmod& fdm_volume,
    const FastDivmod& fdm_num_windows,
    const FastDivmod& fdm_ws_w,
    const FastDivmod& fdm_ws_h,
    const FastDivmod& fdm_windows_w,
    const FastDivmod& fdm_windows_h,
    int positions,
    int D, int H, int W, int C,
    int ws_d, int ws_h, int ws_w,
    int ss_d, int ss_h, int ss_w) {
    constexpr int kVecElems = (kVecBytes >= static_cast<int>(sizeof(scalar_t)))
        ? (kVecBytes / static_cast<int>(sizeof(scalar_t)))
        : 1;
    const int C_vec = C / kVecElems;
    const int sub_warp = select_subwarp(C_vec > 0 ? C_vec : 1);

#define CRUMPET_DISPATCH_SUBWARP(SW)                                                   \
    case SW:                                                                            \
        launch_kernel<scalar_t, Shifted, kVecBytes, SW>(                                \
            output, x, fdm_volume, fdm_num_windows,                                     \
            fdm_ws_w, fdm_ws_h, fdm_windows_w, fdm_windows_h,                           \
            positions, D, H, W, C,                                                      \
            ws_d, ws_h, ws_w,                                                           \
            ss_d, ss_h, ss_w);                                                          \
        break

    switch (sub_warp) {
        CRUMPET_DISPATCH_SUBWARP(1);
        CRUMPET_DISPATCH_SUBWARP(2);
        CRUMPET_DISPATCH_SUBWARP(4);
        CRUMPET_DISPATCH_SUBWARP(8);
        CRUMPET_DISPATCH_SUBWARP(16);
        default:
            CRUMPET_DISPATCH_SUBWARP(32);
    }

#undef CRUMPET_DISPATCH_SUBWARP
}

template <typename scalar_t, bool Shifted>
void launch_shift_partition(
    torch::Tensor& output,
    const torch::Tensor& x,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w) {
    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int W = static_cast<int>(x.size(3));
    const int C = static_cast<int>(x.size(4));
    const int volume = static_cast<int>(ws_d * ws_h * ws_w);
    const int num_windows = static_cast<int>((D / ws_d) * (H / ws_h) * (W / ws_w));
    const int positions = B * num_windows * volume;

    const FastDivmod fdm_volume(volume);
    const FastDivmod fdm_num_windows(num_windows);
    const FastDivmod fdm_ws_w(static_cast<int>(ws_w));
    const FastDivmod fdm_ws_h(static_cast<int>(ws_h));
    const FastDivmod fdm_windows_w(static_cast<int>(W / ws_w));
    const FastDivmod fdm_windows_h(static_cast<int>(H / ws_h));

    const int bytes_per_pos = C * static_cast<int>(sizeof(scalar_t));
    const int vec_bytes = crumpet::select_vec_bytes(bytes_per_pos);

#define CRUMPET_DISPATCH_VEC(VB)                                                       \
    dispatch_subwarp<scalar_t, Shifted, VB>(                                            \
        output, x, fdm_volume, fdm_num_windows,                                         \
        fdm_ws_w, fdm_ws_h, fdm_windows_w, fdm_windows_h,                               \
        positions, D, H, W, C,                                                          \
        static_cast<int>(ws_d), static_cast<int>(ws_h), static_cast<int>(ws_w),         \
        static_cast<int>(ss_d), static_cast<int>(ss_h), static_cast<int>(ss_w))

    switch (vec_bytes) {
        case 16: CRUMPET_DISPATCH_VEC(16); break;
        case 8:  CRUMPET_DISPATCH_VEC(8);  break;
        case 4:  CRUMPET_DISPATCH_VEC(4);  break;
        default: CRUMPET_DISPATCH_VEC(static_cast<int>(sizeof(scalar_t))); break;
    }

#undef CRUMPET_DISPATCH_VEC
}

}  // namespace

void fused_shift_partition_3d(
    torch::Tensor& output,
    const torch::Tensor& x,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w) {
    const at::cuda::CUDAGuard device_guard(x.device());
    if (output.numel() == 0) {
        return;
    }
    const bool shifted = (ss_d != 0 || ss_h != 0 || ss_w != 0);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "crumpet_fused_shift_partition_3d",
        [&] {
            if (shifted) {
                launch_shift_partition<scalar_t, true>(
                    output, x, ws_d, ws_h, ws_w, ss_d, ss_h, ss_w);
            } else {
                launch_shift_partition<scalar_t, false>(
                    output, x, ws_d, ws_h, ws_w, ss_d, ss_h, ss_w);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_unshift_unpartition_3d_backward(
    torch::Tensor& grad_windows,
    const torch::Tensor& grad_x,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w) {
    fused_shift_partition_3d(
        grad_windows,
        grad_x,
        ws_d,
        ws_h,
        ws_w,
        ss_d,
        ss_h,
        ss_w);
}
