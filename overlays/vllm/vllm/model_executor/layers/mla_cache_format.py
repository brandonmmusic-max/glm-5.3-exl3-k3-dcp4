# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-static NVFP4 MLA cache-format configuration and ABI identity."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

NVFP4_MLA_SCALES_ENV = "VLLM_NVFP4_MLA_SCALES_FILE"
NVFP4_MLA_DYNAMIC_SCALE_ENV = "VLLM_NVFP4_MLA_DYNAMIC_SCALE"
KV_FP8_ROPE_ENV = "KV_FP8_ROPE"
GLM_NOPE_NVFP4_ENV = "VLLM_B12X_GLM_NOPE_NVFP4"


@dataclass(frozen=True)
class Nvfp4MlaCacheFormat:
    """Immutable writer/reader configuration captured at process import."""

    dynamic_scale: bool
    fp8_rope: bool
    scales_file: str
    glm_nope: bool = False

    @classmethod
    def from_env(cls) -> Nvfp4MlaCacheFormat:
        return cls(
            dynamic_scale=os.getenv(NVFP4_MLA_DYNAMIC_SCALE_ENV, "0") == "1",
            fp8_rope=os.getenv(KV_FP8_ROPE_ENV, "0") == "1",
            scales_file=os.getenv(NVFP4_MLA_SCALES_ENV, "").strip(),
            glm_nope=os.getenv(GLM_NOPE_NVFP4_ENV, "0") == "1",
        )

    def validate(self) -> None:
        if self.dynamic_scale and self.scales_file:
            raise ValueError(
                f"{NVFP4_MLA_SCALES_ENV} and "
                f"{NVFP4_MLA_DYNAMIC_SCALE_ENV}=1 are mutually exclusive"
            )
        if self.glm_nope and self.fp8_rope:
            raise ValueError(
                f"{GLM_NOPE_NVFP4_ENV}=1 and {KV_FP8_ROPE_ENV}=1 describe "
                "different NVFP4 record layouts"
            )

    def record_abi(self, cache_dtype: str) -> str:
        """Return an identity suitable for persistent external-cache keys."""
        normalized_dtype = str(cache_dtype).replace("torch.", "")
        if normalized_dtype != "nvfp4_ds_mla":
            return "vllm-default-v1"

        self.validate()
        if not self.dynamic_scale and not self.scales_file:
            # Preserve the existing namespace for every unconfigured/default
            # deployment. Only modes that change the record's scale semantics
            # opt into a new external-cache identity.
            return "vllm-default-v1"

        if self.dynamic_scale:
            # GLM-5.3 NoPE has no RoPE sub-record. Its dynamic record is the
            # 288-byte latent payload plus a 16-byte inline-scale trailer.
            # Backends that are neither NoPE nor FP8-RoPE reject this mode
            # during attention initialization.
            layout = "fp8-rope-368" if self.fp8_rope else "nope-304"
            scale_mode = "dynamic-token-v1"
        else:
            if self.fp8_rope:
                layout = "fp8-rope-368"
            elif self.glm_nope:
                # GLM-5.3 stores only the 512-D NoPE latent: 256 packed E2M1
                # bytes plus 32 E4M3 group scales. There is no RoPE sub-record.
                layout = "nope-288"
            else:
                layout = "bf16-rope-432"
            try:
                scale_bytes = Path(self.scales_file).read_bytes()
                scale_digest = hashlib.sha256(scale_bytes).hexdigest()
            except OSError as exc:
                raise ValueError(
                    f"Cannot fingerprint {NVFP4_MLA_SCALES_ENV}={self.scales_file!r}"
                ) from exc
            scale_version = "v1"
            try:
                scale_payload = json.loads(scale_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                scale_payload = None
            if isinstance(scale_payload, dict) and scale_payload.get("format") == (
                "nvfp4_ds_mla_outer_scale_v2"
            ):
                scale_version = "v2"
            scale_mode = f"static-calibrated-{scale_version}:{scale_digest}"
        return f"nvfp4_ds_mla:{layout}:{scale_mode}"


# All consumers import this one frozen value, so a process cannot configure
# the writer, readers, and external-cache namespace from different env reads.
NVFP4_MLA_CACHE_FORMAT = Nvfp4MlaCacheFormat.from_env()
