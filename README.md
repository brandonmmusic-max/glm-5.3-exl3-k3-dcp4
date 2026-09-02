# Full GLM-5.3 EXL3 K3 on Jovian r10: TP4 + DCP4 + MTP3

This repository packages the source overlays and container recipe used for the
**full GLM-5.3** uniform-K3 EXL3 runtime. It is not the GLM-5.3-Flash model.
The expected checkpoint has 3-bit target experts, K5 MTP78 experts, and a
materialization receipt declaring `runtime_tp: 4`.

The Docker image bakes in the vLLM/B12X overlays that were previously mounted
at launch. It does **not** contain model weights. Mount an already materialized
checkpoint read-only at `/model` and a writable compilation cache at `/cache`.

## Status

- Distribution: custom community derivative
- Qualification: research-only
- Maintenance: ephemeral
- Fresh GPU validation: not run for this publication
- Historical evidence: the source set previously reached server-ready TP4,
  DCP4, MTP3 with B12X A2A and a 368-byte NVFP4 DS MLA record; those older
  observations are not a performance or qualification claim for this image.

## Why DCP4 works

The sparse indexer returns merged **logical token IDs** under DCP4
(`output_physical_slots=False`). The B12X attention path then maps those IDs to
each rank's cache-physical MLA slots and uses the direct A2A/LSE combine path.
The model adapter registers the full `GlmMoeDsaForCausalLM` architecture and
selects the 368-byte NVFP4 DS MLA + FP8 RoPE record.

## Build

```bash
docker build \
  --build-arg RECIPE_COMMIT=$(git rev-parse HEAD) \
  -t verdictai/glm53-exl3-k3:jovian-r10-tp4-dcp4-mtp3-research-v1 .
```

## Run

This is the exact intended serving profile. It requires four compatible NVIDIA
GPUs and a pre-existing checkpoint; it does not download anything.

```bash
docker run --rm --gpus '"device=0,1,2,3"' \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /absolute/path/to/GLM-5.3-EXL3-TR3-uniform-K3-runtime:/model:ro \
  -v /absolute/path/to/cache:/cache \
  verdictai/glm53-exl3-k3:jovian-r10-tp4-dcp4-mtp3-research-v1
```

The launcher fixes TP4, DCP4/A2A, MTP3, `gpu-memory-utilization=0.97`,
`nvfp4_ds_mla`, B12X MoE/attention, and full-decode CUDA graphs. The historical
launcher's ignored `VLLM_DCP_INDEXER_SHARDS`, `VLLM_DCP_QUERY_SPLIT`,
`VLLM_DCP_GLOBAL_TOPK`, and `VLLM_DCP_SHARD_DRAFT` variables are deliberately
absent. For exact replay it retains both cache-interleave flags: the deprecated
`--dcp-kv-cache-interleave-size 64` overrides
`--cp-kv-cache-interleave-size 1`, so the effective DCP4 interleave is **64**.
The successful MTP3 cell used checkpoint index reuse; this package does not
claim generic fused-draft support for interleave 64.

See [PROVENANCE.md](docs/PROVENANCE.md) and the machine-readable image record
under `publication/` for exact identities and limitations.

`tools/materialize_checkpoint.py` is retained for reproducibility, but is not
run by the build or launcher. It can perform range requests for pinned official
non-routed tensors; do not invoke it if downloads are prohibited.
