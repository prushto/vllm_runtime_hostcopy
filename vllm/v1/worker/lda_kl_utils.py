# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KL and blend helpers for LDA (logits differential amplification).

Matches the definition in dormant warmup ``lda_engine._kl_divergence_per_row``:
KL(dormant || base) with p = softmax(dormant), q = softmax(base).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_dormant_base(
    dormant_logits: torch.Tensor, base_logits: torch.Tensor
) -> torch.Tensor:
    """KL(dormant || base) for each row (sequence position batch).

    Args:
        dormant_logits: [N, V]
        base_logits: [N, V]

    Returns:
        Tensor of shape [N] with non-negative KL values.
    """
    log_p = F.log_softmax(dormant_logits, dim=-1)
    log_q = F.log_softmax(base_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.clamp(min=0.0)


def lda_blend_logits(
    dormant_logits: torch.Tensor,
    base_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Amplified logits: dormant + alpha * (dormant - base)."""
    return dormant_logits + alpha * (dormant_logits - base_logits)
