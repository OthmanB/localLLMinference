# Model Files

Model weights are intentionally not stored in this repository. Download the
exact files from their authorized source and verify them before use.

## Expected Qwen3.8 GGUF Files

| Profile | Filename | SHA-256 |
|---|---|---|
| UD Q4, RTX 5090 standalone/reference | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| UD Q5, historical comparison | `Qwen3.8-27B-UD-Q5_K_M.gguf` | `2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd` |

## Other Deployment Artifacts

These are distinct model artifacts, not aliases for UD Q4:

| Artifact | Source | Recorded identity |
|---|---|---|
| ATX IQ4_XS-M | `jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF` | SHA-256 `5cf05ad901dcaa76f41db13a5629146ed882219339377a80e37b12a8528d963b` |
| NVFP4 candidate | `unsloth/Qwen3.8-27B-NVFP4` | Engine-native safetensors; verify the downloaded revision and hash locally |
| NVFP4 candidate | `Inferact/Qwen3.8-27B-NVFP4` | Engine-native safetensors; verify the downloaded revision and hash locally |

The ATX artifact is intended for the pinned llamAmpere v0.3 runtime and its
MTP vocabulary map. The NVFP4 artifacts are for the vLLM/SGLang evaluation
plan. They must not be substituted for the GGUF files in a profile without a
separate compatibility and quality gate.

Verify a downloaded file with:

```bash
sha256sum /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
```

Do not commit model files, partial downloads, tokenizer caches, or access
credentials for model hosting services.
