# Provenance

The immutable base is
`verdictai/glm53-exl3-k3@sha256:2f618ef8c734ce02c8589c55bfdf4f9a79d63d7997378f34737d27574bda538a`.
It supplies the Jovian r10 CUDA/PyTorch/vLLM/B12X runtime and the compiled EXL3
uniform-K3 extension.

The base labels identify these principal source revisions:

- vLLM: `local-inference-lab/vllm` at
  `3d33a5a0f596567d4acfe77dc6a9e0f2d07ada77`
- B12X: `local-inference-lab/b12x` at
  `ccb89b6e444bd9eaa9115b97eb6def2a42dede9c`
- exllamav3: `brandonmmusic-max/exllamav3` at
  `704aefd743b390af4bd0fb429d1906f9b964c7d8`
- NVIDIA PyTorch foundation:
  `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`

Relevant upstream review lines are
[vLLM PR 562](https://github.com/local-inference-lab/vllm/pull/562),
[vLLM PR 560](https://github.com/local-inference-lab/vllm/pull/560), and
[vLLM PR 565](https://github.com/local-inference-lab/vllm/pull/565). This
repository is a reproducible composition/package, not a substitute for those
upstream reviews.

`evidence/overlay-manifest.tsv` records each old base-file hash and the baked
overlay hash. `ABSENT` means the file is introduced by this layer.

The checkpoint identity is intentionally external. The launcher rejects a
mount unless its receipt reports the full GLM-5.3 uniform-K3 contract, including
K5 MTP78 and TP4 materialization. No weights, calibration set, scales file, or
Hugging Face credential is included.

