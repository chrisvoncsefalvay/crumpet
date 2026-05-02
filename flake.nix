# CRUMPET
# Author: Chris von Csefalvay
# Licence: MIT
# Hugging Face kernel: https://hf.co/chrisvoncsefalvay/crumpet

{
  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };
  outputs =
    { self, kernel-builder, ... }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}

