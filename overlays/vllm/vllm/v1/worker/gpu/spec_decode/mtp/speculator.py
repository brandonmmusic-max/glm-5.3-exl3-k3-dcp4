# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

logger = init_logger(__name__)


class MTPSpeculator(AutoRegressiveSpeculator):
    share_mtp_topk_indices: bool = False
    rollback_qsa_interval_starts: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        checkpoint_reuse = bool(
            getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
        )
        reuse_mode = spec_config.mtp_index_reuse
        requested_reuse = (
            checkpoint_reuse if reuse_mode == "checkpoint" else reuse_mode == "on"
        )
        supports_reuse = callable(
            getattr(draft_model.model, "set_skip_topk", None)
        ) and callable(getattr(draft_model.model, "compact_topk_indices", None))
        if requested_reuse and not supports_reuse:
            raise RuntimeError(
                "MTP index reuse was requested but the draft model does not "
                "provide set_skip_topk and compact_topk_indices"
            )
        if supports_reuse:
            # Establish the off state even if this instance explicitly
            # disables reuse; do not inherit stale mutable model state.
            draft_model.model.set_skip_topk(False)
        self.share_mtp_topk_indices = requested_reuse
        logger.info(
            "MTP index reuse resolved to %s (mode=%s, checkpoint=%s)",
            self.share_mtp_topk_indices,
            reuse_mode,
            checkpoint_reuse,
        )
        recycle_mode = spec_config.mtp_recycle_mode
        supports_recycle_mode = bool(
            getattr(draft_model.model, "supports_mtp_recycle_mode", False)
        )
        if recycle_mode != "post_norm" and not supports_recycle_mode:
            raise RuntimeError(
                "MTP recycle mode was requested but the draft model does not "
                "declare supports_mtp_recycle_mode"
            )
        logger.info(
            "MTP hidden-state recycle resolved to %s (supported=%s)",
            recycle_mode,
            supports_recycle_mode,
        )
        logger.info(
            "MTP sampling resolved to %s (rejection=%s)",
            spec_config.draft_sample_method,
            spec_config.rejection_sample_method,
        )
        self.rollback_qsa_interval_starts = callable(
            getattr(draft_model.model, "snapshot_qsa_interval_starts", None)
        ) and callable(
            getattr(draft_model.model, "restore_qsa_interval_starts", None)
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        if self.rollback_qsa_interval_starts:
            self.model.model.snapshot_qsa_interval_starts()
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.rollback_qsa_interval_starts:
            self.model.model.restore_qsa_interval_starts()
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
