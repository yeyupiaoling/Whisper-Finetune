"""Offline strategy-level GRPO tests; no pretrained models/audio downloads."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from transformers import WhisperConfig, WhisperForConditionalGeneration, set_seed

from model.transformer import TransformerConfig, TransformerForConditionalGeneration
from train_distil import distillation_loss
from train_grpo_distill import (
    StrategyDistillationTrainer, corpus_wer, cpu_snapshot, distill_steps,
    evaluate_wer, main, parse_args, strategy_grpo_loss,
)


class ToyTokenizer:
    prefix_tokens = [1]
    pad_token_id = 0

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [" ".join(str(int(token)) for token in sequence if int(token) > 2)
                for sequence in sequences]


@contextlib.contextmanager
def audio_stubs(modules):
    # patch.dict(sys.modules) also removes unrelated lazy imports at exit, which
    # can break PyTorch compiler registries when DeepSpeed is imported again.
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class StrategyDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        set_seed(42)

    def args(self, *extra):
        return parse_args(["--reward_data", "dataset/reward.json", "--device", "cpu", *extra])

    def models(self):
        common = dict(vocab_size=16, num_mel_bins=4, d_model=8,
                      encoder_layers=1, decoder_layers=1,
                      encoder_attention_heads=2, decoder_attention_heads=2,
                      encoder_ffn_dim=16, decoder_ffn_dim=16,
                      max_source_positions=4, max_target_positions=8,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2,
                      decoder_start_token_id=1, dropout=0.0)
        student = TransformerForConditionalGeneration(TransformerConfig(**common))
        teacher = WhisperForConditionalGeneration(WhisperConfig(**common)).eval().requires_grad_(False)
        return student, teacher

    def batch(self):
        return dict(input_features=torch.randn(2, 4, 8),
                    labels=torch.tensor([[3, 2, -100], [4, 5, 2]]))

    def assert_states_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_states_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_states_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_cli_action_space_and_validation(self):
        args = self.args("--ce_weights", "0.2", "0.8", "--num_generations", "3")
        student, teacher = self.models()
        trainer = StrategyDistillationTrainer(student, teacher, args)
        self.assertEqual(trainer.ce_weights, [1.0, 0.2, 0.8])
        for invalid in (["--ce_weights", "1"], ["--ce_weights", "0"],
                        ["--ce_weights", "nan"], ["--ce_weights", "0.5", "0.5"],
                        ["--num_generations", "1"], ["--inner_steps", "0"],
                        ["--sampling_temperature", "0"], ["--beta", "nan"],
                        ["--test_data", "dataset/reward.json"]):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*invalid)

    def test_corpus_wer_is_word_weighted_and_handles_empty(self):
        # One error in five reference words, not a mean of sentence WERs (0.5).
        self.assertAlmostEqual(corpus_wer(["a b c d", "e"], ["A B C D!", "x"]), 0.2)
        self.assertEqual(corpus_wer([""], ["a b"]), 2.0)
        self.assertEqual(corpus_wer([""], [""]), 0.0)
        self.assertEqual(corpus_wer(["a"], [""]), 1.0)
        with self.assertRaises(ValueError):
            corpus_wer([], [])

    def test_grpo_favors_wer_reduction_not_increase(self):
        logits = torch.nn.Parameter(torch.zeros(3))
        actions = torch.tensor([0, 1, 2])
        old = F.log_softmax(logits.detach(), -1)[actions]
        # Action 0 improves WER, action 1 has no effect, action 2 makes it worse.
        rewards = torch.tensor([-0.2, 0.0, 0.2])
        loss = strategy_grpo_loss(logits, actions, old, rewards, old, beta=0, entropy_coef=0)
        loss.backward()
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[2].item(), 0)
        updated = (logits.detach() - 0.1 * logits.grad).softmax(-1)
        self.assertGreater(updated[0].item(), updated[2].item())

    def test_tied_rewards_and_exact_reference_kl(self):
        logits = torch.tensor([1.0, -1.0], requires_grad=True)
        actions = torch.tensor([0, 0])
        ref = torch.full((2,), -torch.log(torch.tensor(2.0)).item())
        logps = logits.log_softmax(-1)
        loss = strategy_grpo_loss(logits, actions, logps[actions].detach(),
                                  torch.zeros(2), ref, beta=0.4, entropy_coef=0)
        torch.testing.assert_close(loss, 0.4 * (logps.exp() * (logps - ref)).sum())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ce_only_skips_teacher(self):
        student, teacher = self.models()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        before = student.lm_head.weight.detach().clone()
        with patch.object(teacher, "forward", side_effect=AssertionError("CE must not call teacher")):
            loss = distill_steps(student, teacher, optimizer, [self.batch()], 1.0, 1.5, 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, student.lm_head.weight))

    def test_fixed_mixture_matches_ce_plus_temperature_kl(self):
        student, teacher = self.models()
        batch = self.batch()
        with torch.no_grad():
            outputs = student(**batch)
            expected = 0.25 * outputs.loss + 0.75 * distillation_loss(
                outputs.logits, teacher(**batch).logits, batch["labels"], 2.0)
        loss = distill_steps(student, teacher, torch.optim.AdamW(student.parameters(), lr=1e-3),
                            [batch], 0.25, 2.0, 1.0)
        self.assertAlmostEqual(loss, expected.item(), places=6)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        batch["labels"].fill_(-100)
        with self.assertRaises(ValueError):
            distill_steps(student, teacher, torch.optim.AdamW(student.parameters()),
                          [batch], 0.25, 2.0, 1.0)

    def test_trials_reset_model_optimizer_rng_and_commit_best(self):
        student = torch.nn.Linear(1, 1, bias=False)
        teacher = torch.nn.Linear(1, 1)
        trainer = StrategyDistillationTrainer(student, teacher, self.args("--num_generations", "3"))
        # Populate momentum first: checking just fresh optimizers misses alias bugs.
        student(torch.ones(1, 1)).sum().backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        baseline = cpu_snapshot(student.state_dict())
        optimizer_baseline = cpu_snapshot(trainer.optimizer.state_dict())
        candidates, optimizer_candidates, noise = [], [], []

        def trial(model, teacher, optimizer, batches, weight, temperature, max_norm):
            self.assert_states_equal(model.state_dict(), baseline)
            self.assert_states_equal(optimizer.state_dict(), optimizer_baseline)
            noise.append(torch.rand(1))
            optimizer.zero_grad(set_to_none=True)
            (model(torch.ones(1, 1)).sum() * weight).backward()
            optimizer.step()
            candidates.append(cpu_snapshot(model.state_dict()))
            optimizer_candidates.append(cpu_snapshot(optimizer.state_dict()))
            return weight

        with patch("train_grpo_distill.distill_steps", side_effect=trial), patch(
                "torch.distributions.Categorical.sample", return_value=torch.tensor([0, 1, 2])):
            metrics = trainer.step([{}], Mock(side_effect=[0.8, 0.6, 0.9, 0.7]))
        torch.testing.assert_close(torch.tensor(metrics["rewards"]), torch.tensor([-0.2, 0.1, -0.1]))
        self.assertEqual(metrics["selected_ce_weight"], 1.0)
        self.assert_states_equal(student.state_dict(), candidates[0])
        self.assert_states_equal(trainer.optimizer.state_dict(), optimizer_candidates[0])
        for value in noise[1:]:
            torch.testing.assert_close(value, noise[0])
        self.assertGreater(metrics["strategy_probabilities"][0], metrics["strategy_probabilities"][1])

    def test_failed_trial_restores_student_and_optimizer(self):
        student, teacher = self.models()
        trainer = StrategyDistillationTrainer(student, teacher, self.args())
        before = cpu_snapshot(student.state_dict())
        optimizer_before = cpu_snapshot(trainer.optimizer.state_dict())
        with self.assertRaises(FloatingPointError):
            trainer.step([self.batch()], Mock(side_effect=[0.8, float("nan")]))
        self.assert_states_equal(student.state_dict(), before)
        self.assert_states_equal(trainer.optimizer.state_dict(), optimizer_before)
        self.assertEqual(trainer.iteration, 0)

    def test_greedy_generation_wer_and_training_mode(self):
        student, _ = self.models()
        processor = SimpleNamespace(tokenizer=ToyTokenizer())
        batch = self.batch()
        student.train()
        with patch.object(student, "generate", return_value=torch.tensor([[1, 3, 2], [1, 4, 2]])) as generate:
            value = evaluate_wer(student, [batch], processor, 3)
        self.assertAlmostEqual(value, 1 / 3)
        self.assertTrue(student.training)
        kwargs = generate.call_args.kwargs
        self.assertNotIn("labels", kwargs)
        self.assertFalse(kwargs["generation_config"].do_sample)
        self.assertEqual(kwargs["decoder_input_ids"].tolist(), [[1], [1]])
        # Also exercise the actual local Transformer's GenerationMixin.
        self.assertTrue(torch.isfinite(torch.tensor(evaluate_wer(student, [batch], processor, 3))))
        self.assertTrue(student.training)

    def test_real_rollout_checkpoint_and_deterministic_resume(self):
        student, teacher = self.models()
        args = self.args("--num_generations", "2", "--inner_steps", "2")
        trainer = StrategyDistillationTrainer(student, teacher, args)
        batches = [self.batch(), self.batch()]
        processor = SimpleNamespace(tokenizer=ToyTokenizer(), save_pretrained=Mock())
        score = lambda model: evaluate_wer(model, [batches[0]], processor, 3)
        metrics = trainer.step(batches, score)
        self.assertEqual(len(metrics["rewards"]), 2)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            trainer.save_checkpoint(directory, processor)
            expected = trainer.step(batches, score)
            expected_student = cpu_snapshot(student.state_dict())
            expected_optimizer = cpu_snapshot(trainer.optimizer.state_dict())
            restored_student = TransformerForConditionalGeneration.from_pretrained(directory)
            resumed = StrategyDistillationTrainer(restored_student, teacher, copy.deepcopy(args))
            resumed.load_checkpoint(directory)
            actual = resumed.step(batches, score)
            self.assertEqual(actual, expected)
            self.assert_states_equal(restored_student.state_dict(), expected_student)
            self.assert_states_equal(resumed.optimizer.state_dict(), expected_optimizer)
            changed = StrategyDistillationTrainer(restored_student, teacher,
                                                  self.args("--ce_weights", "0.1"))
            with self.assertRaises(ValueError):
                changed.load_checkpoint(directory)

    def test_main_loop_writes_metrics_and_checkpoints(self):
        student, teacher = self.models()
        batch = self.batch()
        processor = SimpleNamespace(tokenizer=ToyTokenizer(), save_pretrained=Mock())

        def dataset(**kwargs):
            return [{key: value[i] for key, value in batch.items()} for i in range(2)]

        def collator(*args, **kwargs):
            return lambda rows: {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

        # Exercise orchestration with real tiny models, replacing only external
        # audio I/O and pretrained downloads with in-memory fixtures.
        modules = {"utils.reader": SimpleNamespace(CustomDataset=dataset),
                   "utils.data_utils": SimpleNamespace(DataCollatorSpeechSeq2SeqWithPadding=collator)}
        with tempfile.TemporaryDirectory() as directory, audio_stubs(modules), patch(
                "train_grpo_distill.WhisperProcessor.from_pretrained", return_value=processor), patch(
                "train_grpo_distill.WhisperForConditionalGeneration.from_pretrained", return_value=teacher), patch(
                "train_grpo_distill.build_student", return_value=student), patch.dict(
                "os.environ", {"WORLD_SIZE": "1"}), contextlib.redirect_stdout(io.StringIO()):
            main(["--reward_data", "dataset/reward.json", "--device", "cpu",
                  "--num_iterations", "2", "--num_generations", "2", "--max_new_tokens", "3",
                  "--save_steps", "1", "--eval_steps", "1", "--output_dir", directory])
            output = Path(directory)
            metrics = [json.loads(line) for line in
                       (output / "strategy_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["iteration"] for row in metrics], [1, 2])
            for row in metrics:
                self.assertIn("eval_wer", row)
                self.assertEqual(len(row["sampled_ce_weights"]), 2)
                for reward, after in zip(row["rewards"], row["wer_after"]):
                    self.assertAlmostEqual(reward, after - row["wer_before"], places=6)
            for name in ("checkpoint-best", "checkpoint-1", "checkpoint-2", "checkpoint-final"):
                self.assertTrue((output / name / "strategy_state.pt").is_file())
            state = torch.load(output / "checkpoint-final" / "strategy_state.pt", weights_only=True)
            self.assertEqual(state["iteration"], 2)


if __name__ == "__main__":
    unittest.main()