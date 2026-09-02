ARG BASE_IMAGE=verdictai/glm53-exl3-k3@sha256:2f618ef8c734ce02c8589c55bfdf4f9a79d63d7997378f34737d27574bda538a
FROM ${BASE_IMAGE}

ARG RECIPE_COMMIT

LABEL org.opencontainers.image.title="GLM-5.3 EXL3 K3 TP4 DCP4 MTP3 research runtime" \
      org.opencontainers.image.description="Full GLM-5.3 uniform-K3 EXL3 runtime with baked DCP4, MTP3, B12X, and NVFP4 DS MLA overlays; model weights excluded" \
      org.opencontainers.image.source="https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      org.opencontainers.image.version="jovian-r10-tp4-dcp4-mtp3-research-v1" \
      local-inference.model.family="GLM-5.3 full" \
      local-inference.model.quantization="EXL3 uniform K3 target with K5 MTP78" \
      local-inference.model.update-policy="local-read-only-mount" \
      local-inference.runtime.profile="tp4-dcp4-mtp3-nvfp4-ds-mla" \
      local-inference.runtime.kv-record-bytes="368" \
      local-inference.runtime.status="research-only" \
      local-inference.runtime.base.digest="sha256:2f618ef8c734ce02c8589c55bfdf4f9a79d63d7997378f34737d27574bda538a"

COPY overlays/vllm/ /opt/glm53-flash/vllm/
COPY overlays/b12x/ /opt/glm53-flash/b12x/
COPY runtime/serve-glm53-exl3-k3-dcp4 /usr/local/bin/serve-glm53-exl3-k3-dcp4

RUN chmod 0755 /usr/local/bin/serve-glm53-exl3-k3-dcp4 \
    && /opt/venv/bin/python -m compileall -q \
       /opt/glm53-flash/vllm/vllm \
       /opt/glm53-flash/b12x/b12x

ENTRYPOINT ["/usr/local/bin/serve-glm53-exl3-k3-dcp4"]

