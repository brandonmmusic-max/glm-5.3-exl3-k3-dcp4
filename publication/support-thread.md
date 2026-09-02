# Full GLM-5.3 EXL3 K3 TP4 DCP4 MTP3 research image

Custom research-only Jovian r10 image with the 17-file full GLM-5.3 K3 DCP4/MTP3 composition baked over a digest-pinned compiled EXL3 base; model weights are excluded.

## Identity and status

- Release class: `community-derivative`
- Distribution role: `custom`
- Qualification: `research-only`
- Maintenance: `ephemeral`
- Model family: GLM-5.3 full, GlmMoeDsaForCausalLM
- Image and digest: `verdictai/glm53-exl3-k3:jovian-r10-tp4-dcp4-mtp3-research-v1@sha256:d2c9345ba443b49c41fcd8a30b8ca6b30af7e313cf48b65eb6a7cd1478440399`
- Recommended image/control: `verdictai/glm53-exl3-k3:jovian-r10-k3-fused-runtime-base-v1@sha256:2f618ef8c734ce02c8589c55bfdf4f9a79d63d7997378f34737d27574bda538a`

## Community runbook

- Wiki: https://github.com/local-inference-lab/rtx6kpro @ `f8af0aec0db75183e7164ac6178243e219560390`
- Runbook: [models/glm-5.3-flash.md](https://github.com/local-inference-lab/rtx6kpro/blob/f8af0aec0db75183e7164ac6178243e219560390/models/glm-5.3-flash.md)
- Relationship: Closest architecture-related community runbook only; that runbook covers GLM-5.3-Flash, while this image targets the distinct full GLM-5.3 architecture.

## Based on

- Base image and digest: `verdictai/glm53-exl3-k3:jovian-r10-k3-fused-runtime-base-v1@sha256:2f618ef8c734ce02c8589c55bfdf4f9a79d63d7997378f34737d27574bda538a`
- Base credit: Jovian r10 Local Inference Lab composition with the compiled EXL3 uniform-K3 extension; derived from the pinned NVIDIA PyTorch and voipmonitor runtime lineage documented in PROVENANCE.md.

## Build recipe

- Public recipe: https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4/blob/65a04c8e2716af994894ab0ded294c574950fc16/Dockerfile
- Recipe commit: `65a04c8e2716af994894ab0ded294c574950fc16`
- Complete build command:
```bash
docker build --build-arg RECIPE_COMMIT=65a04c8e2716af994894ab0ded294c574950fc16 -t verdictai/glm53-exl3-k3:jovian-r10-tp4-dcp4-mtp3-research-v1 .
```

## Source commits, PRs, patches, and overlays

- **vLLM Jovian candidate plus full GLM-5.3 DCP4 composition:** https://github.com/local-inference-lab/vllm @ `3d33a5a0f596567d4acfe77dc6a9e0f2d07ada77`; release `N/A`
  - PR: https://github.com/local-inference-lab/vllm/pull/562 @ `97ba04f40bb48273762491749b3f0834ec32ad93` — Add EXL3 Trellis support on Jovian; authors: Derek Yates, Brandon Music, David Young, Martin Vit, Michel Belleau
  - PR: https://github.com/local-inference-lab/vllm/pull/560 @ `8d7f8207bfd95ec9fc225acef9ca7fd97b39dd22` — Handle empty DCP ranks and gather sparse MLA queries; authors: Derek Yates
  - PR: https://github.com/local-inference-lab/vllm/pull/565 @ `d461572be161b8dc1ac37964ccd6c1d5694cc105` — Fuse K3 DCP verification queries; authors: myshytf
  - Patch: https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4/blob/65a04c8e2716af994894ab0ded294c574950fc16/evidence/overlay-manifest.tsv — `sha256:deec02c8bd1f8f0a50f989ad1df83e0b7adb03f79972dd2379c9eabfa62609e4` — Identity manifest for the exact baked overlay composition, including old and new file hashes.; authors: Brandon Music, Local Inference Lab contributors
- **B12X candidate:** https://github.com/local-inference-lab/b12x @ `ccb89b6e444bd9eaa9115b97eb6def2a42dede9c`; release `N/A`
- **ExLlamaV3:** https://github.com/brandonmmusic-max/exllamav3 @ `704aefd743b390af4bd0fb429d1906f9b964c7d8`; release `N/A`

### Package changes
- Bakes the previously bind-mounted 14 vLLM and 3 B12X source files into the image.
- Adds a fail-closed launcher for the full GLM-5.3 K3 target plus K5 MTP78 receipt contract.
- Does not add model weights or a calibration/scales artifact.

### Build arguments
- BASE_IMAGE defaults to the immutable K3 compiled-runtime base digest.
- RECIPE_COMMIT=65a04c8e2716af994894ab0ded294c574950fc16

### Environment defaults
- Dynamic NVFP4 DS MLA scale path; no static scales file is bundled.
- EXL3 Trellis prefill block M 64 and small-M split-K enabled.
- B12X direct DCP A2A and PCIe all-reduce enabled with query replication disabled.
- Full decode AOT/CUDA-graph settings enabled; breakable CUDA graphs disabled.

### Entrypoint changes
- Installs /usr/local/bin/serve-glm53-exl3-k3-dcp4 as the image entrypoint.
- Fixes TP4, DCP4 A2A, MTP3 greedy checkpoint reuse, 0.97 memory utilization, and NVFP4 DS MLA.
- Retains both interleave arguments for exact replay; the deprecated DCP value 64 is the effective value.

- Result tree: `dc3b9e5ed802c023192def880015630a22ee09e9`
- Integration patch: `N/A`

## Changes from the base image

### Inherited
- CUDA 13.3, PyTorch 2.13, NCCL 2.31.2, vLLM/B12X runtime, and compiled EXL3 K3 extension from the pinned base.
- The base includes broader Jovian support and stale inherited GLM-5.3-Flash labels.

### Introduced
- Full GLM-5.3 model adapter, logical-to-physical DCP4 sparse MLA mapping, DCP A2A/LSE path, MTP3 plumbing, and fused K3/K5 route-pack source overlays.
- Accurate OCI identity labels for full GLM-5.3, uniform-K3 target, K5 MTP78, TP4/DCP4/MTP3, and a 368-byte NVFP4 DS MLA record.
- Portable launcher with four runtime-ignored DCP environment variables removed.

### Compatibility impact
- Requires four compatible NVIDIA GPUs, a receipt-bearing TP4 materialized checkpoint mounted read-only at /model, and a writable /cache mount.
- The image is intentionally specialized to full GLM-5.3 K3/K5 and is not a GLM-5.3-Flash image.
- DCP1 is not advertised; retained evidence did not establish that path.

## Tested configuration

### Published baked image exact-profile GPU runtime
- Hardware: Not tested
- Topology: Not tested
- Power/clocks: Not tested
- Driver/runtime: Not tested; CUDA Not tested; PyTorch Not tested; NCCL Not tested
- Engine: Not tested
- Model/quant: Not tested; Not tested
- Parallelism: Not tested
- KV/speculation: Not tested; Not tested
- Graph/scheduler: Not tested; Not tested
- Cache/JIT: Not tested
- Launch command:
```bash
Not tested
```

## Validation results

### Commands
- `Not tested`

### Results
- **other / Fresh runtime startup, correctness, stability, and performance validation:** not-tested. Conditions: No model launch or GPU test was authorized for this publication.. Measurement: Not tested. Result: Not tested. Conclusion: The image is research-only and must not be treated as qualified.. Evidence: N/A (`N/A`).

## Performance claims

- N/A

## Known limitations

- The baked image has not received a fresh GPU startup after replacing host bind mounts with image layers.
- Historical runtime evidence used a different launcher hash and cannot qualify this publication.
- Dynamic NVFP4 scales are enabled; no separately calibrated static MLA scale artifact is included.
- The effective DCP cache interleave is 64 because the retained deprecated argument overrides cp interleave 1.

## Untested configurations

- Fresh server readiness and /v1/models identity check
- LLM decode benchmark, acceptance, prefill, stability, and long-context behavior
- DCP1 and any topology other than TP4/DCP4/MTP3

## Unsupported configurations

- GLM-5.3-Flash checkpoints, checkpoints without the exact materialization receipt, CPU serving, and automatic model downloads.

## Support and issue routing

- Support owner: Brandon Music
- Contact: Use the public GitHub support issue.
- Support commitment: `ephemeral`
- Support thread: https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4/issues/1
- Thread status: `active`
- Issue tracker: https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4/issues
- Upstream escalation: Keep reports in issue 1 with immutable image digest, hardware and topology, checkpoint receipt identity, complete launch command, and logs. Escalate upstream only after a minimal reproducer identifies the responsible source change.
- Superseded by: `N/A`

## Publication record

- Machine-readable record: https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4/main/publication/image-record.json
- Main-channel link: N/A
- Automated listing: `not-applicable`
- Maintainer approval: N/A
