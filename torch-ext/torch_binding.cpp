// CRUMPET: fused 3D shifted-window kernels for efficient transformers
// SPDX-License-Identifier: MIT

#include <torch/library.h>

#include "registration_select.h"
#include "torch_binding.h"

namespace {
const std::vector<at::Tag> kCrumpetOpTags = {
    at::Tag::pt2_compliant_tag,
    at::Tag::needs_fixed_stride_order,
};
}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "compute_attn_mask_3d(Tensor! output, int D, int H, int W, int ws_d, int ws_h, int ws_w, int ss_d, int ss_h, int ss_w) -> ()",
        kCrumpetOpTags);
    ops.def(
        "fused_shift_partition_3d(Tensor! output, Tensor x, int ws_d, int ws_h, int ws_w, int ss_d, int ss_h, int ss_w) -> ()",
        kCrumpetOpTags);
    ops.def(
        "fused_shift_partition_3d_backward(Tensor! grad_x, Tensor grad_windows, int B, int D, int H, int W, int C, int ws_d, int ws_h, int ws_w, int ss_d, int ss_h, int ss_w) -> ()",
        kCrumpetOpTags);
    ops.def(
        "fused_unshift_unpartition_3d(Tensor! output, Tensor windows, int B, int D, int H, int W, int C, int ws_d, int ws_h, int ws_w, int ss_d, int ss_h, int ss_w) -> ()",
        kCrumpetOpTags);
    ops.def(
        "fused_unshift_unpartition_3d_backward(Tensor! grad_windows, Tensor grad_x, int ws_d, int ws_h, int ws_w, int ss_d, int ss_h, int ss_w) -> ()",
        kCrumpetOpTags);

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("compute_attn_mask_3d", torch::kCUDA, &compute_attn_mask_3d);
    ops.impl("fused_shift_partition_3d", torch::kCUDA, &fused_shift_partition_3d);
    ops.impl("fused_shift_partition_3d_backward", torch::kCUDA, &fused_shift_partition_3d_backward);
    ops.impl("fused_unshift_unpartition_3d", torch::kCUDA, &fused_unshift_unpartition_3d);
    ops.impl("fused_unshift_unpartition_3d_backward", torch::kCUDA, &fused_unshift_unpartition_3d_backward);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

