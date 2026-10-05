"""GRPO over discrete distillation strategies, NOT transcript tokens.

Actions are CE alone (lambda=1) and fixed mixtures
	lambda * CE(student, labels) + (1-lambda) * T**2 * KL(teacher || student).
Each iteration samples G actions with replacement from a learned categorical
policy. All trials start from identical student AND AdamW states, use the same
training batches/dropout seed, and are scored with greedy WER on the same probe
batches. Only the lowest-WER sampled trial is committed (even if all get worse).
The policy is a global, learned distribution, not an audio-conditioned network.

The reported reward is exactly WER_after - WER_before. Lower is better, so GRPO
uses NEGATED, group-normalized rewards. Maximizing the raw difference would
reward degradation. One on-policy update is made per group; there is no replay.

Example::

	python train_grpo_distill.py --teacher_model openai/whisper-small \
		--train_data dataset/train.json --reward_data dataset/reward.json \
		--test_data dataset/test.json --ce_weights 0.25 0.5 0.75 \
		--num_generations 4 --inner_steps 5 --num_iterations 1000

Use a disjoint reward split: it trains the strategy policy and must not be the
final test set. WER is word-level (whitespace tokenization), NOT CER; Chinese
text needs word segmentation upstream for a meaningful word-level metric.
This loop uses single-process FP32 training. Rollouts are sequential to avoid
G model replicas on GPU; baseline/best snapshots are kept on CPU. Checkpoints
contain the student, processor, policy, both optimizers and RNG state. Resume
with --resume_from_checkpoint (not a Seq2SeqTrainer checkpoint).
"""

import argparse
import copy
import functools
import json
import math
import os
import random
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed

from train_distil import build_student, distillation_loss
from utils.utils import add_arguments, print_arguments


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__,
									 formatter_class=argparse.RawDescriptionHelpFormatter)
	add = functools.partial(add_arguments, argparser=parser)
	for name, default in [
		("train_data", "dataset/train.json"), ("test_data", "dataset/test.json"),
		("teacher_model", "openai/whisper-small"), ("teacher_adapter", None),
		("student_model", None), ("output_dir", "output/grpo-distillation"),
		("language", "Chinese"), ("task", "transcribe"),
		("augment_config_path", None), ("resume_from_checkpoint", None),
		("device", "cuda" if torch.cuda.is_available() else "cpu"),
	]:
		add(name, type=str, default=default, help=name)
	parser.add_argument("--reward_data", required=True,
						help="Disjoint probe split used to train the strategy policy")
	parser.add_argument("--ce_weights", type=float, nargs="+", default=[0.25, 0.5, 0.75],
						help="Fixed CE weights in (0, 1); CE-only (1) is always included")
	for name, default in [
		("num_generations", 4), ("num_iterations", 1000), ("inner_steps", 1),
		("reward_batches", 1), ("per_device_train_batch_size", 4),
		("per_device_eval_batch_size", 4), ("eval_steps", 100), ("save_steps", 100),
		("max_new_tokens", 128), ("student_d_model", 256),
		("student_encoder_layers", 6), ("student_decoder_layers", 4),
		("student_attention_heads", 4), ("student_ffn_dim", 1024), ("seed", 42),
	]:
		add(name, type=int, default=default, help=name)
	for name, default in [
		("learning_rate", 1e-4), ("policy_learning_rate", 1e-2),
		("temperature", 1.5), ("sampling_temperature", 1.0),
		("beta", 0.04), ("epsilon", 0.2), ("entropy_coef", 0.01),
		("weight_decay", 0.0), ("max_grad_norm", 1.0),
		("min_audio_len", 0.5), ("max_audio_len", 30.0), ("student_dropout", 0.1),
	]:
		add(name, type=float, default=default, help=name)
	add("local_files_only", type=bool, default=False, help="Only load local models")
	args = parser.parse_args(argv)
	for name in ("num_iterations", "inner_steps", "reward_batches",
				 "per_device_train_batch_size", "per_device_eval_batch_size",
				 "eval_steps", "save_steps", "max_new_tokens"):
		if getattr(args, name) < 1:
			parser.error(f"{name} must be positive")
	if args.num_generations < 2:
		parser.error("num_generations must be >= 2")
	if any(not 0 < weight < 1 for weight in args.ce_weights):
		parser.error("ce_weights must be in (0, 1); CE-only is added automatically")
	if len(set(args.ce_weights)) != len(args.ce_weights):
		parser.error("ce_weights must not contain duplicates")
	for name in ("learning_rate", "policy_learning_rate", "temperature",
				 "sampling_temperature", "max_grad_norm"):
		if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
			parser.error(f"{name} must be finite and positive")
	for name in ("beta", "entropy_coef", "weight_decay"):
		if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
			parser.error(f"{name} must be finite and nonnegative")
	if not 0 < args.epsilon < 1:
		parser.error("epsilon must be in (0, 1)")
	if args.language is None or args.task not in ("transcribe", "translate"):
		parser.error("Specify a language and task=transcribe or translate")
	if not 0.5 <= args.min_audio_len <= args.max_audio_len <= 30:
		parser.error("Require 0.5 <= min_audio_len <= max_audio_len <= 30")
	paths = [Path(path).resolve() for path in
			 (args.train_data, args.reward_data, args.test_data)]
	if len(set(paths)) != 3:
		parser.error("train_data, reward_data and test_data must be separate splits")
	return args


def corpus_wer(references, hypotheses):
	"""Corpus word errors / reference words; case/punctuation insensitive.

	Empty references contribute insertion errors. An entirely empty reference
	corpus uses denominator 1, preserving a finite penalty for hallucinations.
	"""
	from jiwer import process_words

	if len(references) != len(hypotheses) or not references:
		raise ValueError("WER requires matching, nonempty reference/hypothesis lists")

	def normalize(text):
		text = unicodedata.normalize("NFKC", text).casefold()
		return " ".join("".join(c for c in text
								if not unicodedata.category(c).startswith("P")).split())

	errors = words = 0
	for ref, hyp in zip(references, hypotheses):
		ref, hyp = normalize(ref), normalize(hyp)
		words += len(ref.split())
		if ref:
			result = process_words(ref, hyp)
			errors += result.substitutions + result.deletions + result.insertions
		else:
			errors += len(hyp.split())
	return errors / max(words, 1)


@torch.no_grad()
def evaluate_wer(student, batches, processor, max_new_tokens, prefix_tokens=None):
	was_training = student.training
	student.eval()
	device = next(student.parameters()).device
	tokenizer = processor.tokenizer
	references, hypotheses = [], []
	prefix = torch.tensor(tokenizer.prefix_tokens if prefix_tokens is None else prefix_tokens,
						  device=device, dtype=torch.long)
	config = copy.deepcopy(student.generation_config)
	config.forced_decoder_ids = None
	config.do_sample = False
	config.num_beams = 1
	config.num_return_sequences = 1
	config.return_dict_in_generate = False
	config.max_new_tokens = max_new_tokens
	config.use_cache = False
	try:
		for batch in batches:
			audio = {k: v.to(device) for k, v in batch.items() if k != "labels"}
			decoder_prefix = prefix[None].expand(batch["labels"].shape[0], -1)
			sequences = student.generate(**audio, decoder_input_ids=decoder_prefix,
										 generation_config=config)
			if not torch.equal(sequences[:, :prefix.numel()], decoder_prefix):
				raise RuntimeError("Student generation did not preserve the decoder prefix")
			labels = batch["labels"].masked_fill(batch["labels"].eq(-100), tokenizer.pad_token_id)
			references.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
			hypotheses.extend(tokenizer.batch_decode(sequences[:, prefix.numel():],
													  skip_special_tokens=True))
		return corpus_wer(references, hypotheses)
	finally:
		student.train(was_training)


def strategy_grpo_loss(logits, actions, old_logps, rewards, reference_logps,
					   beta=0.04, epsilon=0.2, entropy_coef=0.01):
	"""One discrete action per rollout; raw WER deltas are costs to minimize."""
	advantages = (rewards.mean() - rewards) / (rewards.std(unbiased=False) + 1e-4)
	logps = F.log_softmax(logits, dim=-1)
	ratio = (logps[actions] - old_logps.detach()).exp()
	objective = torch.minimum(ratio * advantages.detach(),
							  ratio.clamp(1 - epsilon, 1 + epsilon) * advantages.detach())
	# The action space is small: use exact categorical KL to the frozen initial
	# (uniform) strategy policy, not transcript-token KL or a noisy KL estimator.
	probs = logps.exp()
	kl = (probs * (logps - reference_logps.detach())).sum()
	entropy = -(probs * logps).sum()
	return -objective.mean() + beta * kl - entropy_coef * entropy


def distill_steps(student, teacher, optimizer, batches, ce_weight, temperature, max_grad_norm):
	student.train()
	teacher.eval()
	device = next(student.parameters()).device
	losses = []
	for batch in batches:
		inputs = {k: v.to(device) for k, v in batch.items()}
		labels = inputs["labels"]
		if not labels.ne(-100).any():
			raise ValueError("A distillation batch has no supervised target tokens")
		optimizer.zero_grad(set_to_none=True)
		logits = student(**inputs, use_cache=False, return_dict=True).logits
		ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
							 labels.reshape(-1), ignore_index=-100)
		loss = ce
		if ce_weight != 1.0:
			with torch.no_grad():
				teacher_logits = teacher(**inputs, use_cache=False, return_dict=True).logits
			loss = ce_weight * ce + (1 - ce_weight) * distillation_loss(
				logits, teacher_logits, labels, temperature)
		if not torch.isfinite(loss):
			raise FloatingPointError("Non-finite student distillation loss")
		loss.backward()
		torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm, error_if_nonfinite=True)
		optimizer.step()
		losses.append(loss.detach().item())
	optimizer.zero_grad(set_to_none=True)
	return sum(losses) / len(losses)


def cpu_snapshot(value):
	"""Copy model/optimizer state without retaining mutable tensor aliases."""
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().clone()
	if isinstance(value, dict):
		return {key: cpu_snapshot(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return type(value)(cpu_snapshot(item) for item in value)
	return copy.deepcopy(value)


def rng_state():
	numpy_state = np.random.get_state()
	return dict(torch=torch.get_rng_state(), python=random.getstate(),
				numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
				cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
	torch.set_rng_state(state["torch"])
	random.setstate(state["python"])
	numpy_state = state["numpy"]
	np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
						 *numpy_state[2:]))
	if state["cuda"] and torch.cuda.is_available():
		torch.cuda.set_rng_state_all(state["cuda"])


class StrategyDistillationTrainer:
	"""Sequential strategy trials with isolated student and optimizer states."""

	def __init__(self, student, teacher, args):
		self.student = student
		self.teacher = teacher.eval().requires_grad_(False)
		self.args = args
		self.ce_weights = [1.0, *args.ce_weights]
		self.optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate,
										   weight_decay=args.weight_decay)
		# Only this small vector is the GRPO policy. Student gradients come
		# exclusively from the selected CE/KL distillation strategy.
		self.policy_logits = torch.nn.Parameter(torch.zeros(len(self.ce_weights)))
		self.policy_optimizer = torch.optim.Adam([self.policy_logits], lr=args.policy_learning_rate)
		self.reference_logps = F.log_softmax(torch.zeros_like(self.policy_logits), dim=-1)
		self.iteration = 0
		self.best_eval_wer = float("inf")
		self.data_rng = torch.Generator().manual_seed(args.seed)

	def step(self, train_batches, score):
		"""score(student) must use the same held-out examples for every call."""
		cfg = self.args
		logits = self.policy_logits / cfg.sampling_temperature
		distribution = torch.distributions.Categorical(logits=logits)
		actions = distribution.sample((cfg.num_generations,))
		old_logps = distribution.log_prob(actions).detach()
		before = float(score(self.student))
		if not math.isfinite(before):
			raise FloatingPointError("Non-finite baseline WER")
		baseline = cpu_snapshot(self.student.state_dict())
		optimizer_baseline = cpu_snapshot(self.optimizer.state_dict())
		best_state = best_optimizer = None
		after_values, losses = [], []
		best_index = 0
		trial_rng = rng_state()
		try:
			for index, action in enumerate(actions.tolist()):
				self.student.load_state_dict(baseline)
				# load_state_dict can reuse CPU optimizer tensors: deepcopy is
				# necessary even when the student itself runs on CPU.
				self.optimizer.load_state_dict(copy.deepcopy(optimizer_baseline))
				restore_rng(trial_rng)
				losses.append(distill_steps(
					self.student, self.teacher, self.optimizer, train_batches,
					self.ce_weights[action], cfg.temperature, cfg.max_grad_norm))
				after = float(score(self.student))
				if not math.isfinite(after):
					raise FloatingPointError("Non-finite post-distillation WER")
				after_values.append(after)
				if best_state is None or after < after_values[best_index]:
					best_index = index
					best_state = cpu_snapshot(self.student.state_dict())
					best_optimizer = cpu_snapshot(self.optimizer.state_dict())
			rewards = torch.tensor(after_values, dtype=torch.float32) - before
			policy_loss = strategy_grpo_loss(
				logits, actions, old_logps, rewards, self.reference_logps,
				cfg.beta, cfg.epsilon, cfg.entropy_coef)
			if not torch.isfinite(policy_loss):
				raise FloatingPointError("Non-finite strategy policy loss")
			self.policy_optimizer.zero_grad(set_to_none=True)
			policy_loss.backward()
			self.policy_optimizer.step()
			self.student.load_state_dict(best_state)
			self.optimizer.load_state_dict(best_optimizer)
		except Exception:
			self.student.load_state_dict(baseline)
			self.optimizer.load_state_dict(optimizer_baseline)
			raise
		finally:
			# Candidate order/count must not change subsequent data sampling or
			# augmentation RNG. Action sampling already advanced the global RNG.
			restore_rng(trial_rng)
		self.iteration += 1
		return dict(iteration=self.iteration, wer_before=before, wer_after=after_values,
					rewards=rewards.tolist(), sampled_actions=actions.tolist(),
					sampled_ce_weights=[self.ce_weights[i] for i in actions.tolist()],
					selected_ce_weight=self.ce_weights[actions[best_index].item()],
					selected_wer=after_values[best_index], distillation_losses=losses,
					policy_loss=policy_loss.item(), ce_weights=self.ce_weights,
					strategy_probabilities=F.softmax(
						self.policy_logits.detach() / cfg.sampling_temperature, -1).tolist())

	def save_checkpoint(self, directory, processor):
		directory = Path(directory)
		directory.mkdir(parents=True, exist_ok=True)
		self.student.save_pretrained(directory)
		processor.save_pretrained(directory)
		torch.save(dict(
			iteration=self.iteration, best_eval_wer=self.best_eval_wer,
			args=vars(self.args), ce_weights=self.ce_weights,
			policy_logits=self.policy_logits.detach(), reference_logps=self.reference_logps,
			student_optimizer=cpu_snapshot(self.optimizer.state_dict()),
			policy_optimizer=self.policy_optimizer.state_dict(), rng=rng_state(),
			data_rng=self.data_rng.get_state(),
		), directory / "strategy_state.pt")

	def load_checkpoint(self, directory):
		# The student weights/config/processor are restored by build_student.
		state = torch.load(Path(directory) / "strategy_state.pt", map_location="cpu", weights_only=True)
		# Changing rollout semantics makes saved optimizer/policy states invalid.
		mutable = {"num_iterations", "output_dir", "resume_from_checkpoint", "student_model",
				   "device", "local_files_only", "eval_steps", "save_steps"}
		for name, value in vars(self.args).items():
			if name not in mutable and state["args"].get(name) != value:
				raise ValueError(f"Resume argument differs from checkpoint: {name}")
		if state["ce_weights"] != self.ce_weights:
			raise ValueError("Cannot change the strategy action space on resume")
		with torch.no_grad():
			self.policy_logits.copy_(state["policy_logits"])
		self.reference_logps = state["reference_logps"]
		self.optimizer.load_state_dict(state["student_optimizer"])
		self.policy_optimizer.load_state_dict(state["policy_optimizer"])
		self.iteration = state["iteration"]
		self.best_eval_wer = state["best_eval_wer"]
		self.data_rng.set_state(state["data_rng"])
		restore_rng(state["rng"])


def main(argv=None):
	args = parse_args(argv)
	if int(os.environ.get("WORLD_SIZE", "1")) != 1:
		raise ValueError("Strategy trials currently support single-process training only")
	print_arguments(args)
	set_seed(args.seed)
	# Keep audio/augmentation dependencies out of loss-only imports and --help.
	from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
	from utils.reader import CustomDataset

	processor = WhisperProcessor.from_pretrained(
		args.teacher_model, language=args.language, task=args.task,
		no_timestamps=True, local_files_only=args.local_files_only)
	# CustomDataset can mutate tokenizer language when reading individual rows.
	# Freeze the requested generation prefix before any dataset access.
	prefix_tokens = tuple(processor.tokenizer.prefix_tokens)
	teacher = WhisperForConditionalGeneration.from_pretrained(
		args.teacher_model, local_files_only=args.local_files_only)
	if args.teacher_adapter:
		from peft import PeftModel
		teacher = PeftModel.from_pretrained(teacher, args.teacher_adapter,
											is_trainable=False).merge_and_unload()
	student = build_student(args, teacher.config, processor)
	if args.max_new_tokens + len(processor.tokenizer.prefix_tokens) > student.config.max_target_positions:
		raise ValueError("max_new_tokens plus decoder prefix exceeds student decoder capacity")
	teacher.config.use_cache = False
	teacher.to(args.device).eval().requires_grad_(False)
	student.to(args.device)
	collator = DataCollatorSpeechSeq2SeqWithPadding(
		processor, decoder_start_token_id=student.config.decoder_start_token_id)
	datasets = []
	for path, augment in ((args.train_data, args.augment_config_path),
						  (args.reward_data, None), (args.test_data, None)):
		dataset = CustomDataset(data_list_path=path, processor=processor,
								language=args.language, timestamps=False,
								min_duration=args.min_audio_len, max_duration=args.max_audio_len,
								augment_config_path=augment)
		if not len(dataset):
			raise ValueError(f"Empty dataset after duration filtering: {path}")
		datasets.append(dataset)
	train_data, reward_data, eval_data = datasets
	trainer = StrategyDistillationTrainer(student, teacher, args)
	if args.resume_from_checkpoint:
		trainer.load_checkpoint(args.resume_from_checkpoint)
	output = Path(args.output_dir)
	output.mkdir(parents=True, exist_ok=True)

	def sample_batches(dataset, count, batch_size):
		# Random batches with replacement; a dedicated checkpointed RNG avoids
		# costly replay/skip of audio decoding when resuming this outer loop.
		return [collator([dataset[i] for i in torch.randint(
			len(dataset), (batch_size,), generator=trainer.data_rng).tolist()])
				for _ in range(count)]

	def validation_batches():
		for start in range(0, len(eval_data), args.per_device_eval_batch_size):
			yield collator([eval_data[i] for i in range(
				start, min(start + args.per_device_eval_batch_size, len(eval_data)))])

	print(f"Strategy CE weights: {trainer.ce_weights}; reward = WER_after - WER_before (minimize)")
	print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
	for _ in range(trainer.iteration, args.num_iterations):
		train_batches = sample_batches(train_data, args.inner_steps, args.per_device_train_batch_size)
		probes = sample_batches(reward_data, args.reward_batches, args.per_device_eval_batch_size)
		metrics = trainer.step(train_batches, lambda model: evaluate_wer(
			model, probes, processor, args.max_new_tokens, prefix_tokens))
		if trainer.iteration % args.eval_steps == 0 or trainer.iteration == args.num_iterations:
			metrics["eval_wer"] = evaluate_wer(
				student, validation_batches(), processor, args.max_new_tokens, prefix_tokens)
			if metrics["eval_wer"] < trainer.best_eval_wer:
				trainer.best_eval_wer = metrics["eval_wer"]
				trainer.save_checkpoint(output / "checkpoint-best", processor)
		print(json.dumps(metrics, ensure_ascii=False), flush=True)
		with (output / "strategy_metrics.jsonl").open("a", encoding="utf-8") as stream:
			stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
		if trainer.iteration % args.save_steps == 0:
			trainer.save_checkpoint(output / f"checkpoint-{trainer.iteration}", processor)
	trainer.save_checkpoint(output / "checkpoint-final", processor)


if __name__ == "__main__":
	main()
