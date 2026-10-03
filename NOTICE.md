# Notices

Copyright 2026 Haverbex. Cookbook code is licensed under Apache-2.0.

The model weights are not included. The model's applicable weight and source-data terms are independent of this cookbook's code license. The selected research model follows its own non-commercial research conditions; this repository does not grant a commercial model license.

The inference runtime uses PyTorch, torchvision, timm, Transformers, safetensors, Pillow, NumPy and SentencePiece under their respective licenses. Its architecture uses FastViT-T8 and a multilingual MiniLM text encoder; retain the applicable Apple model and upstream text-encoder notices with a distributed model bundle.

`assets/kitchen.png` is a procedural scene produced for this project's demonstrations. `assets/banner.png` is the project's existing ShoreCrab artwork. No benchmark dataset images or game ROMs are distributed here.

The native C++ graph uses GGML from llama.cpp (MIT); its convolution graph construction follows GGML's corresponding implementations. See [GGML's notice](LICENSES/ggml-MIT.txt). The vLLM multimodal processor follows the raw-input processor pattern in vLLM's `terratorch.py` (Apache-2.0, copyright 2025 the vLLM team and IBM). Both engines are external pinned source dependencies; their source trees are not vendored here.
