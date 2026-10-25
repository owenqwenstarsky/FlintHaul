# Longhaul MoE loading

Longhaul mode keeps the non-expert model weights resident and loads routed MoE expert slices from the GGUF file as they are needed. The expert cache has a fixed budget and uses least-recently-used replacement independently for each layer.

This mode is intended for running a model whose full expert weights do not fit in memory. It trades throughput and latency for a smaller resident model allocation.

## Usage

```sh
llama-cli \
    --model /path/to/flint.gguf \
    --longhaul \
    --longhaul-cache 8 \
    --n-gpu-layers 0
```

`--longhaul-cache` is the expert cache budget in GiB. It is required when `--longhaul` is used. `--longhaul` is shorthand for `--load-mode longhaul`. The same options are accepted by `llama-server`, and the environment variables are `LLAMA_ARG_LONGHAUL` and `LLAMA_ARG_LONGHAUL_CACHE`.

The command above runs every repeating layer on CPU and is supported on macOS, Linux, and Windows. On macOS, an all-Metal mode is also available by using `--n-gpu-layers 99` instead. Longhaul does not silently change device placement: an unsupported GPU or mixed CPU/GPU repeating-layer placement fails with guidance to use `--n-gpu-layers 0`.

Longhaul may reduce `--ubatch-size` so that every expert selected by one graph segment can be present in the cache at the same time. The effective value is logged during context creation. If one token selects more experts than the cache has slots, the routed MoE computation is split into multiple stages and the partial results are summed. This permits smaller caches at the cost of additional graph work.

The normal startup warmup is skipped automatically in longhaul mode, as is the server's startup sequence-removal probe and the device-memory fitting step. Routed expert weights are not read until the first real decode.

## Current scope

Longhaul currently requires:

- all repeating layers on CPU, or all repeating layers on Metal
- the Flint architecture
- text generation without embeddings or LoRA adapters

CPU mode is available wherever the CPU backend is supported. Metal mode requires macOS. Longhaul does not restrict the GGUF quantization type; individual tensor types must still be supported by the selected compute backend.

MTP/speculative decoding, tensor validation during loading, vocabulary-only loading, `prism.hadamard` models, mixed CPU/GPU repeating-layer placement, and the GPU MoE cache (`--moe-cache-mib`) are not supported together with longhaul. Both single-file and split GGUF models are supported; routed expert tensors are read from the shard that owns each tensor.

The cache budget covers the compact routed-expert tensors. It does not include dense weights, attention weights, the shared expert, the KV cache, graph allocations, or temporary staging buffers. CPU expert caches use the standard directly writable tensor layout so individual slots can be replaced; optimized CPU buffer selection continues to apply to all non-streamed weights.

Per-expert scale tensors (for example NVFP4 scales) stay resident and are looked up with the router's expert IDs, while the expert matmuls read the cache slots chosen for those IDs.

File-cache control is best effort and is outside the explicit cache budget. macOS disables caching for streamed files where supported, and Linux advises the kernel to discard completed reads. Windows may retain file data in its system cache. Windows reads from one GGUF shard are serialized to preserve positional read correctness, while POSIX systems retain concurrent `pread` operations.

This implementation synchronizes at each routed MoE layer to discover the selected experts, populate missing cache slots, and continue execution with cache-local expert IDs. Storage speed and expert reuse therefore have a large effect on generation speed.

Expert IDs are planned as a batch at each synchronization point. Experts already needed by that batch are protected from eviction, duplicate IDs are loaded only once, and independent expert slices are read concurrently where the platform supports positional reads. CPU and shared Metal buffers are populated directly; private Metal buffers use a staged fallback.

Flint keeps its shared expert resident and streams only the routed expert banks (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`). Flint selects six of 256 routed experts per token, so a cache with at least six slots executes each MoE layer in one stage. Smaller budgets down to one slot per layer are supported by splitting the selected experts into multiple stages. The slot size is the sum of the three expert tensors' per-expert bytes across all MoE layers, and the load log reports the resulting slot count.

## Benchmarking prompt processing

`llama-bench` accepts the longhaul load mode and cache budget:

```sh
llama-bench \
    --model /path/to/flint.gguf \
    --load-mode longhaul \
    --longhaul-cache 8 \
    --n-gpu-layers 0 \
    --n-prompt 2048 \
    --n-gen 0
```

Use `--no-warmup --repetitions 1` in separate processes to measure a cold expert cache. Leave warmup enabled to measure steady-state cache reuse.
