# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LDA KL and blend (CPU, no GPU).

Run from repo root or dev container:

  python -m pytest test/basic_func/test_lda_kl_utils.py -q
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.v1.worker.lda_kl_utils import kl_divergence_dormant_base, lda_blend_logits


def _kl_reference_dormant_base(dormant_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """Duplicate of dormant warmup lda_engine._kl_divergence_per_row for parity."""
    log_p = F.log_softmax(dormant_logits, dim=-1)
    log_q = F.log_softmax(base_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.clamp(min=0.0)


def test_kl_zero_when_logits_identical() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    kl = kl_divergence_dormant_base(x, x)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=0.0, rtol=0.0)


def test_kl_matches_reference_implementation() -> None:
    torch.manual_seed(1)
    d = torch.randn(3, 64)
    b = torch.randn(3, 64)
    k1 = kl_divergence_dormant_base(d, b)
    k2 = _kl_reference_dormant_base(d, b)
    assert torch.allclose(k1, k2)


def test_kl_non_negative() -> None:
    d = torch.randn(2, 32)
    b = torch.randn(2, 32)
    kl = kl_divergence_dormant_base(d, b)
    assert (kl >= -1e-6).all()


def test_lda_blend_algebra() -> None:
    torch.manual_seed(2)
    d = torch.randn(2, 16)
    b = torch.randn(2, 16)
    alpha = 2.5
    out = lda_blend_logits(d, b, alpha)
    expected = d + alpha * (d - b)
    assert torch.allclose(out, expected)


def test_lda_blend_same_logits_equals_identity() -> None:
    x = torch.randn(1, 8)
    for alpha in (0.0, 0.5, 3.0):
        out = lda_blend_logits(x, x, alpha)
        assert torch.allclose(out, x)


if __name__ == "__main__":
    test_kl_zero_when_logits_identical()
    test_kl_matches_reference_implementation()
    test_kl_non_negative()
    test_lda_blend_algebra()
    test_lda_blend_same_logits_equals_identity()
    print("OK: all lda_kl_utils checks passed")
