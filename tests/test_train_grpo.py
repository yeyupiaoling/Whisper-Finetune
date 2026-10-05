"""Offline numerical tests: python -m unittest discover -s tests -p test_train_grpo.py.

Load the mathematical helpers through AST so tests need only PyTorch, not the
audio/PEFT stack or downloaded checkpoints.
"""
import ast
from pathlib import Path
import unittest
import unicodedata

import torch
import torch.nn.functional as F


source = Path(__file__).resolve().parents[1] / "train_grpo.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
           and node.name in {"normalize_text", "edit_distance", "transcript_reward",
                             "completion_mask", "grpo_loss"}]
trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "SpeechGRPOTrainer")
scoring = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
               and node.name == "token_logps")
scoring.decorator_list = []
namespace = dict(torch=torch, F=F, unicodedata=unicodedata)
exec(compile(ast.Module(body=helpers + [scoring], type_ignores=[]), str(source), "exec"), namespace)


class GRPOTests(unittest.TestCase):
    def test_rewards(self):
        reward = namespace["transcript_reward"]
        self.assertEqual(reward("你好，世界！", "你好世界"), 0)
        self.assertEqual(reward("你好", "你"), -0.5)
        self.assertEqual(reward("a b", "a c d", "wer"), -1)
        self.assertEqual(reward("", "abc"), -3)
        self.assertEqual(reward("Ａ B", "a b"), 0)

    def test_shared_eos_pad(self):
        mask = namespace["completion_mask"](torch.tensor([[4, 2, 2, 2], [4, 5, 6, 7]]), 2, 2)
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True] * 4])
        mask = namespace["completion_mask"](torch.tensor([[2, 0, 0]]), 2, 0)
        self.assertEqual(mask.tolist(), [[True, False, False]])

    def test_advantage_gradient_and_padding(self):
        for loss_type in ("grpo", "cispo"):
            logps = torch.full((4, 3), -2.0, requires_grad=True)
            mask = torch.tensor([[True, True, False]] * 4)
            loss, kl = namespace["grpo_loss"](
                logps, logps.detach(), logps.detach(), torch.tensor([0., -1., 2., 2.]),
                mask, 2, loss_type=loss_type)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(kl.item(), 0)
            loss.backward()
            self.assertLess(logps.grad[0, 0].item(), 0)
            self.assertGreater(logps.grad[1, 0].item(), 0)
            self.assertEqual(logps.grad[:, 2].abs().sum().item(), 0)
            self.assertEqual(logps.grad[2:].abs().sum().item(), 0)

    def test_kl_and_clipping(self):
        logps = torch.tensor([[-1.], [-3.]], requires_grad=True)
        old = torch.full_like(logps, -2.)
        ref = torch.full_like(logps, -2.)
        loss, kl = namespace["grpo_loss"](
            logps, old, ref, torch.tensor([1., 0.]), torch.ones_like(logps, dtype=torch.bool), 2)
        self.assertGreater(kl.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logps.grad).all())

    def test_decoder_alignment_and_backward(self):
        from types import SimpleNamespace
        logits = torch.randn(2, 5, 10, requires_grad=True)
        sequences = torch.tensor([[1, 3, 4, 6, 7, 2], [1, 3, 4, 8, 2, 2]])

        def model(**kwargs):
            torch.testing.assert_close(kwargs["decoder_input_ids"], sequences[:, :-1])
            self.assertFalse(kwargs["use_cache"])
            return SimpleNamespace(logits=logits)

        actual = namespace["token_logps"](model, {}, sequences, 3)
        expected = logits[:, 2:].log_softmax(-1).gather(-1, sequences[:, 3:, None]).squeeze(-1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(logits.grad[:, :2].abs().sum().item(), 0)
        self.assertGreater(logits.grad[:, 2:].abs().sum().item(), 0)

    def test_tiny_whisper_rollout_and_policy_gradient(self):
        try:
            from transformers import WhisperConfig, WhisperForConditionalGeneration
            from transformers.generation import GenerationMixin
        except ImportError:
            self.skipTest("Optional Transformers dependency unavailable")
        torch.set_num_threads(1)
        model = WhisperForConditionalGeneration(WhisperConfig(
            vocab_size=16, num_mel_bins=4, d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_source_positions=4, max_target_positions=12,
            pad_token_id=2, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
            suppress_tokens=[], begin_suppress_tokens=[])).eval()
        audio = {"input_features": torch.randn(1, 4, 8).repeat_interleave(2, 0)}
        prefix = torch.tensor([[1, 3, 4], [1, 3, 4]])
        with torch.no_grad():
            sequences = GenerationMixin.generate(
                model, **audio, decoder_input_ids=prefix, do_sample=True,
                top_k=0, max_new_tokens=4, use_cache=True)
        torch.testing.assert_close(sequences[:, :3], prefix)
        logps = namespace["token_logps"](model, audio, sequences, 3)
        mask = namespace["completion_mask"](sequences[:, 3:], 2, 2)
        loss, _ = namespace["grpo_loss"](
            logps, logps.detach(), None, torch.tensor([0., -1.]), mask, 2)
        loss.backward()
        self.assertTrue(torch.isfinite(model.proj_out.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
