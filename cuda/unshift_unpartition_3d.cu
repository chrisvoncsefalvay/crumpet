// CRUMPET: fused 3D shifted-window inverse partition kernel.
// SPDX-License-Identifier: MIT
//
// Mirror of shift_partition_3d.cu — see that file for the full design
// rationale. The inverse mapping decomposes the output position
// (b, d, h, w) directly via FastDivmod and reads the corresponding
// source from the windowed buffer. All optimisations carry over:
// 16-byte LDG.E.CI.128 / STG.E.128, sub-warp packing, FastDivmod, and
// the optional load / store phase split that keeps `kMaxILP` outstanding
// loads in flight per thread.

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
constexpr int kBlockSize = 128;
constexpr int kBlocksPerSm = 8;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
// See shift_partition_3d.cu — empirically tuned ILP budget.
constexpr int kMaxILP = 1;

template <typename scalar_t, bool Shifted, int kVecBytes, int kSubWarpSize>
__global__ void __launch_bounds__(kBlockSize, kBlocksPerSm)
unshift_unpartition_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ windows,
    const FastDivmod fdm_W,
    const FastDivmod fdm_H,
    const FastDivmod fdm_D,
    const FastDivmod fdm_ws_w,
    const FastDivmod fdm_ws_h,
    const FastDivmod fdm_ws_d,
    const int positions,
    const int D, const int H, const int W, const int C,
    const int num_windows, const int volume,
    const int windows_h, const int windows_w,
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

    const int sub_lane = threadIdx.x;
    const int row = threadIdx.y;
    const int pos = blockIdx.x * kPosPerBlock + row;
    const bool active = (pos < positions);

    int src_base = 0;
    int out_base = 0;
    if (active && sub_lane == 0) {
        int tmp_dh, w;
        fdm_W.divmod(tmp_dh, w, pos);
        int tmp_d, h;
        fdm_H.divmod(tmp_d, h, tmp_dh);
        int b, d;
        fdm_D.divmod(b, d, tmp_d);

        int part_d = d, part_h = h, part_w = w;
        if constexpr (Shifted) {
            part_d -= ss_d;
            part_h -= ss_h;
            part_w -= ss_w;
            part_d += (part_d < 0) ? D : 0;
            part_h += (part_h < 0) ? H : 0;
            part_w += (part_w < 0) ? W : 0;
        }

        int win_d, loc_d;
        fdm_ws_d.divmod(win_d, loc_d, part_d);
        int win_h, loc_h;
        fdm_ws_h.divmod(win_h, loc_h, part_h);
        int win_w, loc_w;
        fdm_ws_w.divmod(win_w, loc_w, part_w);

        const int win_flat = (win_d * windows_h + win_h) * windows_w + win_w;
        const int local = (loc_d * fdm_ws_h.divisor + loc_h) * fdm_ws_w.divisor + loc_w;
        src_base = ((b * num_windows + win_flat) * volume + local) * C;
        out_base = pos * C;
    }

    const int lane_in_warp = (row * kSubWarpSize + sub_lane) & (kWarpSize - 1);
    const int leader_lane = (lane_in_warp / kSubWarpSize) * kSubWarpSize;
    src_base = __shfl_sync(kFullMask, src_base, leader_lane);
    out_base = __shfl_sync(kFullMask, out_base, leader_lane);

    if (!active) return;

    // C-dim copy. See shift_partition_3d.cu for the full rationale.
    const int C_vec = C / kVecElems;
    const scalar_t* src_pos = windows + src_base;
    scalar_t* out_pos = output + out_base;

    using Vec = typename VecOps<kVecBytes>::vec_t;
    Vec buf[kMaxILP];

    #pragma unroll
    for (int u = 0; u < kMaxILP; ++u) {
        const int v = sub_lane + u * kSubWarpSize;
        if (v < C_vec) {
            buf[u] = VecOps<kVecBytes>::template load<scalar_t>(
                src_pos + v * kVecElems);
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

    #pragma unroll 1
    for (int v = sub_lane + kMaxILP * kSubWarpSize; v < C_vec; v += kSubWarpSize) {
        VecCopy<kVecBytes>::template copy<scalar_t>(
            out_pos + v * kVecElems, src_pos + v * kVecElems);
    }

    const int tail_start = C_vec * kVecElems;
    #pragma unroll 1
    for (int c = tail_start + sub_lane; c < C; c += kSubWarpSize) {
        out_pos[c] = src_pos[c];
    }
}

template <typename scalar_t, bool Shifted, int kVecBytes, int kSubWarpSize>
inline void launch_kernel(
    torch::Tensor& output,
    const torch::Tensor& windows,
    const FastDivmod& fdm_W,
    const FastDivmod& fdm_H,
    const FastDivmod& fdm_D,
    const FastDivmod& fdm_ws_w,
    const FastDivmod& fdm_ws_h,
    const FastDivmod& fdm_ws_d,
    int positions,
    int D, int H, int W, int C,
    int num_windows, int volume,
    int windows_h, int windows_w,
    int ss_d, int ss_h, int ss_w) {
    constexpr int kPosPerBlock = kBlockSize / kSubWarpSize;
    const dim3 threads(kSubWarpSize, kPosPerBlock);
    const int blocks = (positions + kPosPerBlock - 1) / kPosPerBlock;
    unshift_unpartition_kernel<scalar_t, Shifted, kVecBytes, kSubWarpSize><<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        windows.data_ptr<scalar_t>(),
        fdm_W, fdm_H, fdm_D,
        fdm_ws_w, fdm_ws_h, fdm_ws_d,
        positions,
        D, H, W, C,
        num_windows, volume, windows_h, windows_w,
        ss_d, ss_h, ss_w);
}

// See shift_partition_3d.cu for the heuristic rationale.
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
    const torch::Tensor& windows,
    const FastDivmod& fdm_W,
    const FastDivmod& fdm_H,
    const FastDivmod& fdm_D,
    const FastDivmod& fdm_ws_w,
    const FastDivmod& fdm_ws_h,
    const FastDivmod& fdm_ws_d,
    int positions,
    int D, int H, int W, int C,
    int num_windows, int volume,
    int windows_h, int windows_w,
    int ss_d, int ss_h, int ss_w) {
    constexpr int kVecElems = (kVecBytes >= static_cast<int>(sizeof(scalar_t)))
        ? (kVecBytes / static_cast<int>(sizeof(scalar_t)))
        : 1;
    const int C_vec = C / kVecElems;
    const int sub_warp = select_subwarp(C_vec > 0 ? C_vec : 1);

#define CRUMPET_DISPATCH_SUBWARP(SW)                                                   \
    case SW:                                                                            \
        launch_kernel<scalar_t, Shifted, kVecBytes, SW>(                                    \
            output, windows,                                                            \
            fdm_W, fdm_H, fdm_D,                                                        \
            fdm_ws_w, fdm_ws_h, fdm_ws_d,                                               \
            positions, D, H, W, C,                                                      \
            num_windows, volume, windows_h, windows_w,                                  \
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
void launch_unshift_unpartition(
    torch::Tensor& output,
    const torch::Tensor& windows,
    int64_t B,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t C,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w) {
    const int positions = static_cast<int>(B * D * H * W);
    const int volume = static_cast<int>(ws_d * ws_h * ws_w);
    const int num_windows = static_cast<int>((D / ws_d) * (H / ws_h) * (W / ws_w));
    const int windows_h = static_cast<int>(H / ws_h);
    const int windows_w = static_cast<int>(W / ws_w);

    const FastDivmod fdm_W(static_cast<int>(W));
    const FastDivmod fdm_H(static_cast<int>(H));
    const FastDivmod fdm_D(static_cast<int>(D));
    const FastDivmod fdm_ws_w(static_cast<int>(ws_w));
    const FastDivmod fdm_ws_h(static_cast<int>(ws_h));
    const FastDivmod fdm_ws_d(static_cast<int>(ws_d));

    const int bytes_per_pos = static_cast<int>(C) * static_cast<int>(sizeof(scalar_t));
    const int vec_bytes = crumpet::select_vec_bytes(bytes_per_pos);

#define CRUMPET_DISPATCH_VEC(VB)                                                       \
    dispatch_subwarp<scalar_t, Shifted, VB>(                                            \
        output, windows,                                                                \
        fdm_W, fdm_H, fdm_D,                                                            \
        fdm_ws_w, fdm_ws_h, fdm_ws_d,                                                   \
        positions,                                                                      \
        static_cast<int>(D), static_cast<int>(H),                                       \
        static_cast<int>(W), static_cast<int>(C),                                       \
        num_windows, volume, windows_h, windows_w,                                      \
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

void fused_unshift_unpartition_3d(
    torch::Tensor& output,
    const torch::Tensor& windows,
    int64_t B,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t C,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w) {
    const at::cuda::CUDAGuard device_guard(windows.device());
    if (output.numel() == 0) {
        return;
    }
    const bool shifted = (ss_d != 0 || ss_h != 0 || ss_w != 0);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        windows.scalar_type(),
        "crumpet_fused_unshift_unpartition_3d",
        [&] {
            if (shifted) {
                launch_unshift_unpartition<scalar_t, true>(
                    output, windows, B, D, H, W, C, ws_d, ws_h, ws_w, ss_d, ss_h, ss_w);
            } else {
                launch_unshift_unpartition<scalar_t, false>(
                    output, windows, B, D, H, W, C, ws_d, ws_h, ws_w, ss_d, ss_h, ss_w);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_shift_partition_3d_backward(
    torch::Tensor& grad_x,
    const torch::Tensor& grad_windows,
    int64_t B,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t C,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w) {
    fused_unshift_unpartition_3d(
        grad_x,
        grad_windows,
        B,
        D,
        H,
        W,
        C,
        ws_d,
        ws_h,
        ws_w,
        ss_d,
        ss_h,
        ss_w);
}
