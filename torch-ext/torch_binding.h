// CRUMPET: fused 3D shifted-window kernels for efficient transformers
// SPDX-License-Identifier: MIT
#pragma once

#include <torch/all.h>

void compute_attn_mask_3d(
    torch::Tensor& output,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w);

void fused_shift_partition_3d(
    torch::Tensor& output,
    const torch::Tensor& x,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w);

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
    int64_t ss_w);

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
    int64_t ss_w);

void fused_unshift_unpartition_3d_backward(
    torch::Tensor& grad_windows,
    const torch::Tensor& grad_x,
    int64_t ws_d,
    int64_t ws_h,
    int64_t ws_w,
    int64_t ss_d,
    int64_t ss_h,
    int64_t ss_w);

