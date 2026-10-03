# Native engines

A60 takes an image, a question and 2–16 candidate answers. It produces candidate probabilities and a “None” probability. This interface uses complete A60 inference inside each engine.

| Engine | Integration | Validated configuration |
|---|---|---|
| vLLM | Registered `ShoreCrabA60Model`, multimodal processor and pooling output | macOS Apple Silicon CPU, FP32, one request, tensor parallel size 1 |
| llama.cpp | `llama_shorecrab_eval` added to `libllama`, GGUF weights, GGML graph, `llama-shorecrab` CLI | macOS Apple Silicon CPU, FP32, two compute threads |

The llama.cpp path requires this extension build. Standard upstream `llama-cli`, `llama-server`, and `llama_model_load_from_file` do not recognize A60. The vLLM path uses `LLM.encode(..., pooling_task="plugin")`; the chat-completions endpoint is not its interface. CUDA/Metal, quantization, parallel requests and other engine revisions have not been validated.

[Version pins](../native/versions.json) identify the exact source revisions. [Validation](native-validation.json) records the checks and measured numerical differences. These checks establish implementation agreement, not task accuracy or a latency benchmark.

Both engines expose `--output-mode probabilities` and `--output-mode select`. See [the two complete examples per engine](output-modes.md). The default `full` output retains the original interface.

Fresh traffic-frame verification: [24 native CLI calls and 2 invalid-input checks](driving-depth.md#independent-engine-checks), covering both output modes and candidate-order reversal.

## vLLM

Use a separate Python environment from the cookbook's `serve` extra, since this vLLM revision requires PyTorch 2.13.0.

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install uv

git clone https://github.com/vllm-project/vllm.git /path/to/vllm
git -C /path/to/vllm checkout 5f30fc7031cae49bf51073fc953d419b08f8887c
uv pip install -r /path/to/vllm/requirements/cpu.txt \
  'setuptools-rust>=1.9.0' 'setuptools-scm>=8' 'cmake>=3.26.1' wheel jinja2 timm==1.0.30 sentencepiece
MAX_JOBS=2 VLLM_TARGET_DEVICE=cpu uv pip install --no-build-isolation --no-deps -e /path/to/vllm
uv pip install --no-deps -e .

python -m shorecrab.native.prepare_vllm \
  --bundle /path/to/a60-inference-bundle --output /path/to/a60-vllm

OMP_NUM_THREADS=2 python -m shorecrab.native.vllm_cli \
  --model /path/to/a60-vllm --image assets/kitchen.png \
  --question 'What color is the mug?' --choices white blue red black
```

Input preparation performs tokenization, tiling, normalization and geometry construction. The vLLM worker loads the actual A60 parameters and runs both encoders, visual resampling, candidate fusion and output heads. It has no dependency on a running ShoreCrab HTTP server.

## llama.cpp

Build the pinned source together with the extension:

```bash
git clone https://github.com/ggml-org/llama.cpp.git /path/to/llama.cpp
git -C /path/to/llama.cpp checkout cb7934c52ca8710994b2ecc19775ebefcfdb8d01
cmake -S native/llama_cpp -B build/a60 \
  -DLLAMA_CPP_SOURCE=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BLAS=ON
cmake --build build/a60 --target llama-shorecrab -j 2
```

Convert an authorized inference bundle once in the cookbook's PyTorch 2.10 environment:

```bash
python -m pip install -e '.[serve]' /path/to/llama.cpp/gguf-py
python -m shorecrab.native.convert_gguf \
  --bundle /path/to/a60-inference-bundle --output /path/to/a60-f32.gguf
```

Prepare pixels and token IDs, then execute the model in `libllama`:

```bash
python -m shorecrab.native.prepare_gguf \
  --tokenizer /path/to/a60-inference-bundle/tokenizer \
  --image assets/kitchen.png --question 'What color is the mug?' \
  --choices white blue red black --output /path/to/input.gguf

build/a60/llama-shorecrab /path/to/a60-f32.gguf /path/to/input.gguf /path/to/result.json
cat /path/to/result.json
```

The input file contains pixels, masks, geometry and token IDs. It contains no computed model embeddings. The C++ process performs the full image encoder, BERT encoder and scoring graph without Python, LibTorch or an inference subprocess. `libllama` exports the C API declared in [shorecrab.h](../native/llama_cpp/shorecrab.h).

The output probabilities follow the input candidate order. Values are uncalibrated. Do not interpret relative-depth choices as a metric depth map or generic image scores as the game-specific head results in the technical report.
