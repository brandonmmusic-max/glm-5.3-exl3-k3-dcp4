"""GLM-5.3 full-model host adapter for the R10 sparse B12X path.

The R10 CUDA-specific DeepSeek-V3.2 model injects the sparse MLA backend
directly, but it bypasses the generic MLA DCP query-replication/combine path.
This adapter keeps the generic GLM-MoE-DSA implementation and makes the same
backend selection explicit by overriding the process-local B12X registry
entry before model construction.  It also publishes the physical 368-byte
NVFP4+FP8-RoPE record size to R10's KV allocator; the semantic MLA width
remains 576.
"""

from dataclasses import replace
import os

from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import (
    GlmMoeDsaForCausalLM as _GenericGlmMoeDsaForCausalLM,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)


register_backend(
    AttentionBackendEnum.B12X,
    "vllm.v1.attention.backends.mla.b12x_mla_sparse.B12xMLASparseBackend",
)


_original_get_kv_cache_spec = MLAAttention.get_kv_cache_spec


def _glm53_get_kv_cache_spec(self, vllm_config):
    """Stamp the compact GLM record size without changing semantic geometry."""

    spec = _original_get_kv_cache_spec(self, vllm_config)
    backend = self.get_attn_backend()
    if (
        os.environ.get("KV_FP8_ROPE", "0") == "1"
        and self.kv_cache_dtype == "nvfp4_ds_mla"
        and self.head_size == 576
        and backend.__name__ == "B12xMLASparseBackend"
    ):
        return replace(
            spec,
            state_content_bytes=368,
            model_version="glm_fp8_rope",
        )
    return spec


MLAAttention.get_kv_cache_spec = _glm53_get_kv_cache_spec


class GlmMoeDsaForCausalLM(_GenericGlmMoeDsaForCausalLM):
    """Generic GLM model with the R10 B12X sparse-MLA backend selected."""


__all__ = ["GlmMoeDsaForCausalLM"]
