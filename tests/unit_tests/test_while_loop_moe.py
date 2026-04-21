# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.moe import (
    _run_experts_chunked_while_loop,
    _run_experts_grouped_mm,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestWhileLoopChunkedExperts(unittest.TestCase):
    """Numerical equivalence tests for _run_experts_chunked_while_loop."""

    NUM_EXPERTS = 4
    DIM = 64
    HIDDEN_DIM = 128

    def _make_weights(self):
        """Expert weights with trunc_normal_ init matching production models."""
        w1 = torch.empty(
            self.NUM_EXPERTS, self.HIDDEN_DIM, self.DIM, device="cuda"
        )
        w2 = torch.empty(
            self.NUM_EXPERTS, self.DIM, self.HIDDEN_DIM, device="cuda"
        )
        w3 = torch.empty(
            self.NUM_EXPERTS, self.HIDDEN_DIM, self.DIM, device="cuda"
        )
        nn.init.trunc_normal_(w1, std=0.02)
        nn.init.trunc_normal_(w2, std=0.02)
        nn.init.trunc_normal_(w3, std=0.02)
        return w1, w2, w3

    def _make_inputs(self, tokens_per_expert: list[int]):
        num_tokens_per_expert = torch.tensor(
            tokens_per_expert, dtype=torch.float32, device="cuda"
        )
        total = sum(tokens_per_expert)
        # Uniform [0.1, 1.1] avoids near-zero inputs that land in
        # SiLU's flat region and produce noise-dominated gradients.
        x = torch.rand(total, self.DIM, device="cuda") + 0.1
        return x, num_tokens_per_expert

    def test_while_loop_matches_grouped_mm(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([10, 8, 12, 6])

        expected = _run_experts_grouped_mm(w1, w2, w3, x, num_tpe)
        actual = _run_experts_chunked_while_loop(
            w1, w2, w3, x, num_tpe, chunk_size=16, use_grouped_mm=True,
        )

        self.assertEqual(expected.shape, actual.shape)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_while_loop_chunk_size_one(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([3, 2, 4, 1])

        expected = _run_experts_grouped_mm(w1, w2, w3, x, num_tpe)
        actual = _run_experts_chunked_while_loop(
            w1, w2, w3, x, num_tpe, chunk_size=1, use_grouped_mm=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_while_loop_chunk_equals_total(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([5, 5, 5, 5])

        expected = _run_experts_grouped_mm(w1, w2, w3, x, num_tpe)
        actual = _run_experts_chunked_while_loop(
            w1, w2, w3, x, num_tpe, chunk_size=20, use_grouped_mm=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_while_loop_chunk_larger_than_total(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([2, 3, 1, 4])

        expected = _run_experts_grouped_mm(w1, w2, w3, x, num_tpe)
        actual = _run_experts_chunked_while_loop(
            w1, w2, w3, x, num_tpe, chunk_size=100, use_grouped_mm=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_while_loop_backward_matches(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([4, 6, 3, 7])

        x_ref = x.clone().detach().requires_grad_(True)
        w1_ref = w1.clone().detach().requires_grad_(True)
        w2_ref = w2.clone().detach().requires_grad_(True)
        w3_ref = w3.clone().detach().requires_grad_(True)
        out_ref = _run_experts_grouped_mm(w1_ref, w2_ref, w3_ref, x_ref, num_tpe)
        out_ref.sum().backward()

        x_wl = x.clone().detach().requires_grad_(True)
        w1_wl = w1.clone().detach().requires_grad_(True)
        w2_wl = w2.clone().detach().requires_grad_(True)
        w3_wl = w3.clone().detach().requires_grad_(True)
        out_wl = _run_experts_chunked_while_loop(
            w1_wl, w2_wl, w3_wl, x_wl, num_tpe, chunk_size=8, use_grouped_mm=True,
        )
        out_wl.sum().backward()

        torch.testing.assert_close(
            out_wl, out_ref, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            x_wl.grad, x_ref.grad, atol=1e-5, rtol=1e-5
        )
        # Weight gradients: chunked processing splits each expert's tokens
        # across chunks, changing bf16 accumulation order in dW = x^T @ grad_y.
        # With production-scale init (trunc_normal_ std=0.02), measured max abs
        # diff is ~0.002 over 200 seeds.
        for g_wl, g_ref in [
            (w1_wl.grad, w1_ref.grad),
            (w2_wl.grad, w2_ref.grad),
            (w3_wl.grad, w3_ref.grad),
        ]:
            torch.testing.assert_close(g_wl, g_ref, atol=0.005, rtol=0.01)

    def test_while_loop_some_experts_empty(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([0, 10, 0, 5])

        expected = _run_experts_grouped_mm(w1, w2, w3, x, num_tpe)
        actual = _run_experts_chunked_while_loop(
            w1, w2, w3, x, num_tpe, chunk_size=4, use_grouped_mm=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_while_loop_zero_tokens(self):
        w1, w2, w3 = self._make_weights()
        x, num_tpe = self._make_inputs([0, 0, 0, 0])

        expected = _run_experts_grouped_mm(w1, w2, w3, x, num_tpe)
        actual = _run_experts_chunked_while_loop(
            w1, w2, w3, x, num_tpe, chunk_size=8, use_grouped_mm=True,
        )

        self.assertEqual(actual.shape, expected.shape)


if __name__ == "__main__":
    unittest.main()
