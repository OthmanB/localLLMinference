# Model Files

Model weights are intentionally not stored in this repository. Download the
exact files from their authorized source and verify them before use.

## Expected Qwen3.8 GGUF Files

| Profile | Filename | SHA-256 |
|---|---|---|
| Q4 | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Q5 | `Qwen3.8-27B-UD-Q5_K_M.gguf` | `2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd` |

Verify a downloaded file with:

```bash
sha256sum /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
```

Do not commit model files, partial downloads, tokenizer caches, or access
credentials for model hosting services.
