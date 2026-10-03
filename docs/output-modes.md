# Probabilities and answer selection

A60 evaluates an image, question and a user-defined set of 2–16 answers. The two output modes are views of the same distribution, not separate models or a text-generation mode.

`probabilities[i]` corresponds to `choices[i]`. Candidate probabilities plus `none_probability` sum to one. The values are uncalibrated. Selection compares **all candidates and None**; when None wins, `answer` and `answer_index` are null. Equal maxima follow input order, with None last.

## Python / HTTP

Return the complete distribution:

```python
from shorecrab import CrabClient
from shorecrab.outputs import probability_view

client = CrabClient("http://127.0.0.1:8090")
choices = ["white", "red", "blue", "black"]
result = client.score("assets/kitchen.png", "What color is the mug?", choices)
print(probability_view(choices, result))
```

Select one answer:

```python
from shorecrab import CrabClient

client = CrabClient("http://127.0.0.1:8090")
print(client.choose("assets/kitchen.png", "What color is the mug?",
                    ["white", "red", "blue", "black"]))
```

If you already have the scores, reuse them without another inference request:

```python
from shorecrab.outputs import select_answer
print(select_answer(choices, result))
```

## vLLM

Use the pinned CPU installation and converted bundle described in [Native engines](native-engines.md#vllm). The worker calls `LLM.encode(..., pooling_task="plugin")`; this is not a chat-completions request.

```bash
# Mode 1: all candidate probabilities
OMP_NUM_THREADS=2 python -m shorecrab.native.vllm_cli \
  --model /path/to/a60-vllm --image assets/kitchen.png \
  --question 'What color is the mug?' --choices white red blue black \
  --output-mode probabilities
```

```bash
# Mode 2: the most likely candidate or None
OMP_NUM_THREADS=2 python -m shorecrab.native.vllm_cli \
  --model /path/to/a60-vllm --image assets/kitchen.png \
  --question 'What color is the mug?' --choices white red blue black \
  --output-mode select
```

## llama.cpp

Build the pinned `llama-shorecrab` extension and convert the authorized bundle as described in [Native engines](native-engines.md#llamacpp). Run these commands in the cookbook's PyTorch 2.10 environment with `gguf-py` installed. Temporary input/output files are removed after each successful or failed call.

```bash
# Mode 1: all candidate probabilities
OMP_NUM_THREADS=2 python -m shorecrab.native.llama_cpp_cli \
  --executable build/a60/llama-shorecrab --model /path/to/a60-f32.gguf \
  --tokenizer /path/to/a60-inference-bundle/tokenizer \
  --image assets/kitchen.png --question 'What color is the mug?' \
  --choices white red blue black --output-mode probabilities
```

```bash
# Mode 2: the most likely candidate or None
OMP_NUM_THREADS=2 python -m shorecrab.native.llama_cpp_cli \
  --executable build/a60/llama-shorecrab --model /path/to/a60-f32.gguf \
  --tokenizer /path/to/a60-inference-bundle/tokenizer \
  --image assets/kitchen.png --question 'What color is the mug?' \
  --choices white red blue black --output-mode select
```

Only input preprocessing runs in Python. The executable runs A60 inference through `llama_shorecrab_eval` in `libllama`; there is no external inference service. Stock upstream llama.cpp model loading does not recognize this architecture.

## Read an output correctly

The following is an illustrative output shape, not a measured prediction:

```json
{"choices":["red","blue"],"probabilities":[0.2,0.5],"none_probability":0.3,"calibrated":false}
```

Selecting that distribution returns `blue` with probability `0.5`. For `[0.2, 0.1]` and None `0.7`, selection returns null. Do not discard None or renormalize candidates and call the result the model's original confidence.
