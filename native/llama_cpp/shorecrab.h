// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llama.h"
#ifdef __cplusplus
extern "C" {
#endif
// Complete FP32 A60 inference inside libllama. Returns 0 on success.
// Inputs contain preprocessed image pixels, geometry and token IDs (no embeddings).
LLAMA_API int llama_shorecrab_eval(const char *model_path,const char *input_path,
                                  const char *output_path,const char *debug_prefix);
#ifdef __cplusplus
}
#endif
