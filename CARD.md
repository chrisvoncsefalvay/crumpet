# CRUMPET

## Overview

CRUMPET (CUDA accelerated Roll, Unpartition, Mask and Partition for Efficient Transformers) is a forward-only CUDA + Triton kernel package for the shifted-window mechanics and window-attention compute used by 3D Swin models, especially MONAI Swin UNETR. It fuses cyclic roll, window partition, window reverse, reverse roll, shifted attention mask construction and the window-attention scaled-dot-product compute into a small set of GPU kernels.

## Public API

```python
crumpet.compute_attn_mask_3d(D, H, W, window_size, shift_size, dtype, device)
crumpet.fused_shift_partition_3d(x, window_size, shift_size)
crumpet.fused_unshift_unpartition_3d(windows, B, D, H, W, C, window_size, shift_size)
crumpet.patch_monai_swin_unetr()
crumpet.unpatch_monai_swin_unetr()
```

A Triton FlashAttention2-style fused window-attention kernel is wired in by the MONAI patch and replaces the eager `q @ k.T + bias + mask -> softmax -> attn @ v` chain at inference.

## Supported dtypes

- `torch.float32`
- `torch.float16`
- `torch.bfloat16`

## Supported GPUs

The build file targets SM 8.0, SM 8.6, SM 9.0 and SM 12.0 PTX-compatible devices. Validation in this workspace used an NVIDIA GB10 through PTX JIT.

## Compatibility assumptions

- Python 3.10 or later.
- PyTorch 2.5 or later.
- CUDA 12.1 or later.
- Tensor layout `[B, D, H, W, C]`.
- Raw kernels require spatial dimensions divisible by `window_size`. The MONAI patch handles padding internally.
- The fused window-attention kernel needs `head_dim >= 16` (Triton `tl.dot` constraint) and runs at inference only.

## Measured performance

Measured on NVIDIA GB10, PyTorch 2.10.0+cu130, Triton 3.6, CUDA 13.0, fp16. Result JSON files are included under `benchmarks/results/`.

End-to-end real BTCV inference (`img0025`, MONAI Swin UNETR bundle, 96^3 ROI, 20 iterations):

| Path | Mean per inference | vs baseline |
| --- | ---: | ---: |
| Eager PyTorch | 5218 ms | 1.00x |
| Eager + `--compile` | 2449 ms | 2.13x |
| CRUMPET patched | 3082 ms | 1.69x |
| CRUMPET patched + `--compile` | 1910 ms | 2.73x |

Per-kernel speedups vs the eager reference (BTCV-shape, fp16):

| Kernel | Speedup |
| --- | ---: |
| `fused_shift_partition_3d` (BTCV stage 0) | 2.90x |
| `fused_unshift_unpartition_3d` (BTCV stage 0) | 2.74x |
| `compute_attn_mask_3d` (D=98) | 22.4x |
| `fused_swin_attention` (BTCV stage 0) | 7.4x |

Dice on the reference label is within fp16 noise: baseline 0.16146, patched 0.16144 (delta -2.5e-5).

## Real model demo

`python -m crumpet.demo_btcv` supports the MONAI Swin UNETR BTCV bundle and a user-supplied abdominal CT NIfTI. The checked-in `btcv_bundle_demo.json` is a real BTCV run using `img0025` from Synapse `RawData.zip`. The synthetic smoke result is kept separately as `btcv_bundle_demo_synthetic.json`.

## Known limitations

- NVIDIA CUDA only.
- ROCm and Metal are untested.
- Channels-last spatial layout `[B, D, H, W, C]` is required.
- The fused window-attention kernel is forward-only and falls back to eager when `attn_drop > 0` or `head_dim < 16`.
- Dynamic shapes inside one compiled graph are not promised.
- The package is not a diagnostic medical device.

## Safety and medical-use disclaimer

CRUMPET changes model execution mechanics, not model training or medical validation. Validate outputs for each deployment context. Do not use it for diagnosis without the required clinical, regulatory and quality-system controls.

## Citation

If you use CRUMPET, please cite:

```bibtex
@misc{crumpet2026,
  title        = {CRUMPET: fused 3D shifted-window kernels for efficient transformers},
  author       = {von Csefalvay, Chris},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/chrisvoncsefalvay/crumpet}}
}
```

CRUMPET follows the shifted-window mechanics described by Liu et al. 2021 for Swin Transformer and the 3D Swin UNETR implementation by Hatamizadeh et al. 2022. Microsoft Swin Transformer provides 2D fused window-process CUDA prior art.

## Author

I'm [Chris von Csefalvay](https://chrisvoncsefalvay.com), an AI researcher specialising in post-training, and the author of _[Post-Training: A Practical Guide for AI Engineers and Developers](https://posttraining.guide)_ (No Starch Press, 2026). I also write [Post-Slop](https://postslop.substack.com), a periodic diatribe about AI, and what it's doing for society. You can also find me on [LinkedIn](https://linkedin.com/in/chrisvoncsefalvay) and [X](https://x.com/epichrisis).

## License

MIT. See [LICENSE](LICENSE) in the repository.
