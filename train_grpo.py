"""Whisper GRPO fine-tuning using the dataset conventions from finetune.py.

Example:
    python train_grpo.py --base_model models/whisper-sft --num_generations 4

Each audio has G sampled transcripts. Reward is negative CER (or WER).
One on-policy update is made per rollout; there is no rollout replay. Evaluation
uses supervised loss on the held-out transcripts, so best checkpoints are selected
by eval_loss. Use a merged SFT model or --initial_adapter as the starting policy.
"""

import argparse
import copy
import os
import platform
import unicodedata

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration, WhisperProcessor, set_seed,
)
from transformers.generation import GenerationMixin

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.reader import CustomDataset
from utils.utils import add_arguments, print_arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep the common fine-tuning options without importing a multi-model CLI.
    for name, value_type, default, help_text in [
        ("train_data", str, "dataset/train.json", "Training dataset path"),
        ("test_data", str, "dataset/test.json", "Evaluation dataset path"),
        ("base_model", str, "openai/whisper-tiny", "Whisper model ID or local path"),
        ("output_dir", str, "output/grpo", "Output directory"),
        ("warmup_steps", int, 50, "Warmup steps"),
        ("logging_steps", int, 100, "Logging interval"),
        ("eval_steps", int, 1000, "Evaluation interval"),
        ("save_steps", int, 1000, "Checkpoint interval"),
        ("num_workers", int, 8, "Data loader workers"),
        ("learning_rate", float, 1e-6, "Learning rate"),
        ("min_audio_len", float, 0.5, "Minimum audio duration in seconds"),
        ("max_audio_len", float, 30, "Maximum audio duration in seconds"),
        ("use_adalora", bool, False, "Reserved; only fixed-rank LoRA is supported"),
        ("fp16", bool, True, "Use FP16 training"),
        ("use_8bit", bool, False, "Load model in 8-bit precision"),
        ("timestamps", bool, False, "Must be False for text rewards"),
        ("use_compile", bool, False, "Must be False for rollout generation"),
        ("local_files_only", bool, False, "Only load local model files"),
        ("num_train_epochs", int, 3, "Training epochs"),
        ("language", str, "Chinese", "Whisper transcription language"),
        ("augment_config_path", str, None, "Audio augmentation configuration"),
        ("resume_from_checkpoint", str, None, "Checkpoint to resume training from"),
        ("per_device_train_batch_size", int, 1, "Training audio batch size"),
        ("per_device_eval_batch_size", int, 8, "Evaluation batch size"),
        ("gradient_accumulation_steps", int, 1, "Gradient accumulation steps"),
        ("push_to_hub", bool, False, "Push model to Hugging Face Hub"),
        ("hub_model_id", str, None, "Hub repository ID"),
        ("save_total_limit", int, 10, "Maximum saved checkpoints"),
    ]:
        add_arguments(name, value_type, default, help_text, parser)
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04, help="Reference KL weight")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--loss_type", choices=["grpo", "cispo"], default="grpo")
    parser.add_argument("--reward_metric", choices=["cer", "wer"], default="cer")
    parser.add_argument("--initial_adapter", default=None,
                        help="Optional SFT adapter; also used for the fixed reference on resume")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.base_model = args.base_model or "openai/whisper-tiny"
    if args.timestamps:
        parser.error("Text CER/WER rewards require --timestamps False")
    if args.language is None:
        parser.error("Specify --language so that every rollout has a fixed decoder prefix")
    if args.use_adalora:
        parser.error("GRPO currently supports fixed-rank LoRA; use --use_adalora False")
    if args.use_compile:
        parser.error("Rollout generation currently requires --use_compile False")
    if args.num_generations < 2 or args.max_new_tokens < 1:
        parser.error("num_generations must be >= 2 and max_new_tokens must be positive")
    if args.temperature <= 0 or args.beta < 0 or not 0 < args.epsilon < 1 or args.epsilon_high <= 0:
        parser.error("Invalid temperature, beta, epsilon or epsilon_high")
    for name in ("per_device_train_batch_size", "per_device_eval_batch_size",
                 "gradient_accumulation_steps", "num_train_epochs", "logging_steps",
                 "eval_steps", "save_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.save_steps % args.eval_steps:
        parser.error("save_steps must be a multiple of eval_steps for best-model selection")
    if platform.system() == "Windows":
        args.num_workers = 0
    return args


def normalize_text(text):
    """Ignore case/punctuation; keep word boundaries for WER."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c for c in text if not unicodedata.category(c).startswith("P")).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def transcript_reward(reference, hypothesis, metric="cer"):
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if metric == "wer":
        reference, hypothesis = reference.split(), hypothesis.split()
    else:
        reference, hypothesis = "".join(reference.split()), "".join(hypothesis.split())
    # Do not cap error rates: excessive insertions should continue to be penalized.
    return -edit_distance(reference, hypothesis) / max(len(reference), 1)


def completion_mask(tokens, eos_token_id, pad_token_id):
    """Include the first EOS, even when EOS and PAD share an ID (Whisper)."""
    eos = tokens.eq(eos_token_id)
    mask = (eos.cumsum(dim=1) - eos.long()).eq(0)
    if pad_token_id != eos_token_id:
        mask &= tokens.ne(pad_token_id)
    return mask


def grpo_loss(logps, old_logps, ref_logps, rewards, mask, num_generations,
              beta=0.04, epsilon=0.2, loss_type="grpo", epsilon_high=5.0):
    groups = rewards.reshape(-1, num_generations)
    advantages = ((groups - groups.mean(1, keepdim=True)) /
                  (groups.std(1, keepdim=True, unbiased=False) + 1e-4)).reshape(-1)
    ratio = (logps - old_logps).exp()
    if loss_type == "cispo":
        objective = ratio.clamp(max=epsilon_high).detach() * advantages[:, None] * logps
    else:
        objective = torch.minimum(ratio * advantages[:, None],
                                  ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None])
    kl = torch.zeros_like(logps)
    if ref_logps is not None:
        delta = ref_logps - logps
        kl = delta.exp() - delta - 1
    lengths = mask.sum(1).clamp_min(1)
    loss = ((-objective + beta * kl).masked_fill(~mask, 0).sum(1) / lengths).mean()
    return loss, (kl.detach().masked_fill(~mask, 0).sum(1) / lengths).mean()


class SpeechGRPOTrainer(Seq2SeqTrainer):
    def __init__(self, *args, grpo_args, reference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grpo_args = grpo_args
        self.reference_model = reference_model
        # Loss is already averaged per audio group, not per reference-label token.
        self.model_accepts_loss_kwargs = False
        self._grpo_metrics = []

    @staticmethod
    def token_logps(model, audio, sequences, prefix_length):
        outputs = model(**audio, decoder_input_ids=sequences[:, :-1], use_cache=False)
        logits = outputs.logits[:, prefix_length - 1:, :].float()
        targets = sequences[:, prefix_length:]
        return -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Stable supervised validation also allows Trainer's best-model selection.
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        if return_outputs:
            raise ValueError("GRPO training does not return supervised logits")
        cfg = self.grpo_args
        policy = self.accelerator.unwrap_model(model)
        tokenizer = self.processing_class.tokenizer
        labels = inputs["labels"].masked_fill(inputs["labels"].eq(-100), tokenizer.pad_token_id)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        audio = {k: v.repeat_interleave(cfg.num_generations, dim=0)
                 for k, v in inputs.items() if k != "labels"}
        batch_size = labels.shape[0] * cfg.num_generations
        prefix = list(tokenizer.prefix_tokens)
        decoder_prefix = torch.tensor(prefix, device=labels.device).expand(batch_size, -1)
        generation_config = copy.deepcopy(policy.generation_config)
        generation_config.forced_decoder_ids = None
        generation_config.suppress_tokens = []
        generation_config.begin_suppress_tokens = []
        generation_config.do_sample = True
        generation_config.temperature = cfg.temperature
        generation_config.top_k = 0
        generation_config.top_p = 1.0
        generation_config.num_beams = 1
        generation_config.num_return_sequences = 1
        generation_config.max_new_tokens = cfg.max_new_tokens
        generation_config.return_dict_in_generate = False
        generation_config.language = cfg.language
        generation_config.task = cfg.task
        generation_config.return_timestamps = False
        # Disable dropout for both rollout and differentiable policy scoring.
        model.eval()
        try:
            with torch.no_grad():
                # Use raw seq2seq generation: Whisper's transcription wrapper can
                # strip prefix/EOS tokens and apply long-form segment processing.
                # get_base_model retains the injected, active LoRA layers.
                generation_model = policy.get_base_model()
                sequences = GenerationMixin.generate(generation_model,
                    **audio, decoder_input_ids=decoder_prefix,
                    generation_config=generation_config, use_cache=True)
            if sequences.shape[1] <= len(prefix) or not torch.equal(sequences[:, :len(prefix)], decoder_prefix):
                raise RuntimeError("Generation did not preserve the expected decoder prefix")
            completions = sequences[:, len(prefix):]
            mask = completion_mask(completions, tokenizer.eos_token_id, tokenizer.pad_token_id)
            hypotheses = tokenizer.batch_decode(completions, skip_special_tokens=True)
            rewards = torch.tensor([
                transcript_reward(references[i // cfg.num_generations], text, cfg.reward_metric)
                for i, text in enumerate(hypotheses)
            ], dtype=torch.float32, device=sequences.device)
            ref_logps = None
            if self.reference_model is not None:
                with torch.no_grad():
                    ref_logps = self.token_logps(self.reference_model, audio, sequences, len(prefix))
            logps = self.token_logps(model, audio, sequences, len(prefix))
            # Fresh samples, one update: the behavior policy is the current policy.
            # Detaching gives the on-policy ratio a value of 1 with a policy gradient.
            loss, kl = grpo_loss(
                logps, logps.detach(), ref_logps, rewards, mask, cfg.num_generations,
                cfg.beta, cfg.epsilon, cfg.loss_type, cfg.epsilon_high)
            self._grpo_metrics.append(torch.stack([
                rewards.mean(), kl, mask.sum(1).float().mean()]).detach())
            return loss
        finally:
            model.train()

    def log(self, logs, start_time=None):
        if "loss" in logs and self._grpo_metrics:
            stats = torch.stack(self._grpo_metrics).mean(0)
            stats = self.accelerator.gather(stats[None]).mean(0).tolist()
            logs.update(dict(zip(("reward", "reference_kl", "completion_length"), stats)))
            self._grpo_metrics.clear()
        super().log(logs, start_time)


def main(argv=None):
    args = parse_args(argv)
    print_arguments(args)
    set_seed(args.seed)
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task=args.task,
        no_timestamps=True, local_files_only=args.local_files_only)
    dataset_kwargs = dict(processor=processor, language=args.language, timestamps=False,
                          min_duration=args.min_audio_len, max_duration=args.max_audio_len)
    train_dataset = CustomDataset(data_list_path=args.train_data,
                                  augment_config_path=args.augment_config_path, **dataset_kwargs)
    eval_dataset = CustomDataset(data_list_path=args.test_data, **dataset_kwargs)
    if not len(train_dataset) or not len(eval_dataset):
        raise ValueError("Training/evaluation dataset is empty; check paths and duration filters")
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model.rstrip("/\\")))
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs, fp16=args.fp16,
        logging_steps=args.logging_steps, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        optim="adamw_torch", report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers, remove_unused_columns=False,
        label_names=["labels"], ddp_find_unused_parameters=False,
        push_to_hub=args.push_to_hub, hub_model_id=args.hub_model_id, seed=args.seed,
    )
    # One complete policy/reference per process; avoid automatic cross-GPU sharding.
    model_kwargs = dict(local_files_only=args.local_files_only,
                        device_map={"": str(training_args.device)})
    if args.use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    def load_base():
        base = WhisperForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
        if base.config.model_type != "whisper":
            raise ValueError(f"Expected whisper, got {base.config.model_type}")
        base.config.pad_token_id = processor.tokenizer.pad_token_id
        base.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        base.config.use_cache = False
        return base

    model = load_base()
    max_positions = getattr(model.config, "max_target_positions", None)
    prefix_length = len(processor.tokenizer.prefix_tokens)
    if max_positions and args.max_new_tokens + prefix_length > max_positions:
        raise ValueError("max_new_tokens plus decoder prefix exceeds model decoder capacity")
    reference_model = None
    if args.beta:
        reference_model = load_base()
        if args.initial_adapter:
            reference_model = PeftModel.from_pretrained(reference_model, args.initial_adapter)
        reference_model.eval().requires_grad_(False)
    if args.use_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    adapter_path = args.resume_from_checkpoint or args.initial_adapter
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        if any(getattr(config, "peft_type", None) != "LORA" for config in model.peft_config.values()):
            raise ValueError("Only fixed-rank LoRA adapters are supported")
    else:
        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        target_modules = [name for name in
                          ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
                          if name in module_names]
        if not target_modules:
            raise ValueError("No supported LoRA target modules found")
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.0, bias="none"))
    trainer = SpeechGRPOTrainer(
        model=model, args=training_args, grpo_args=args, reference_model=reference_model,
        train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=processor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, decoder_start_token_id=model.config.decoder_start_token_id))
    if trainer.is_world_process_zero():
        model.print_trainable_parameters()
    # Keep Trainer's native checkpoint loader to restore adapters AND optimizer/RNG state.
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(os.path.join(output_dir, "checkpoint-final"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
