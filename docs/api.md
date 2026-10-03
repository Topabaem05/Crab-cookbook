# Decision API

`POST /v1/decisions` takes one image, one question and 2–16 distinct candidate strings.

```json
{
  "image": "data:image/png;base64,...",
  "question": "What color is the mug?",
  "choices": ["white", "red", "blue", "black"]
}
```

`image` accepts a PNG, JPEG or WebP data URI, up to 16 MiB. Arbitrary file paths and remote image URLs are not accepted by the server. The complete image must fit the configured pixel and tiling budgets; oversized inputs are rejected rather than partially read. Overlong text is rejected rather than truncated.

The response preserves input candidate order in `probabilities`. `none_probability` is the additional probability for “none of these”; all probabilities together sum to one. `answer_index` and `answer` are null when that output wins. `calibrated: false` means the values are scores from the model, not a validated confidence or safety guarantee. `usage` records the input size and number of local tiles. `latency_ms` covers the local inference call.

Send `Authorization: Bearer $CRAB_API_KEY` when the server has that environment variable configured. Binding outside localhost requires a key. Put TLS and deployment access controls in front of a remote service.

`GET /health` returns the loaded model's readiness. A successful health request does not measure model accuracy.

| Status | Meaning |
|---|---|
| 200 | Complete decision or ready status |
| 400 | Invalid image, text, candidates or request budget |
| 401 | Missing or invalid configured API key |
| 404 | Unknown route |
| 500 | Inference failed; no partial decision is returned |
