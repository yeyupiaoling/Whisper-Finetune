"""Distil a Whisper teacher into a pretrained Whisper or local audio Transformer.

Example::

    python train_distil.py --teacher_model openai/whisper-small --lambda 0.5
    python train_distil.py --teacher_model openai/whisper-large-v2 --student_model openai/whisper-small
    torchrun --standalone --nproc_per_node=2 train_distil.py --teacher_model openai/whisper-large-v2 --student_model openai/whisper-small

Loss = lambda * student_cross_entropy + (1 - lambda) * temperature_scaled_KL.

Without --student_model, the local Transformer is initialized randomly.
With --student_model, load pretrained weights and auto-detect the architecture.
Teacher/student token-ID mappings and Mel feature dimensions must match.
Use --resume_from_checkpoint to restore Trainer/optimizer/scheduler state.
Multi-GPU training uses torchrun/DDP: one frozen teacher and student per GPU.
Batch sizes are per GPU; effective batch = batch * accumulation * world size.
"""

import argparse
import functools
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
    WhisperProcessor, set_seed,
)

from model.transformer import TransformerConfig, TransformerForConditionalGeneration
from utils.utils import add_arguments, print_arguments


def distillation_loss(student_logits, teacher_logits, labels, temperature=1.5):
    """Token-mean KL(teacher || student), excluding ignored targets."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher/student logits must share sequence length and vocabulary")
    mask = labels.ne(-100)
    if not mask.any():
        return student_logits.sum() * 0.0
    # Compute softmax in float32 even during mixed-precision training.
    student = F.log_softmax(student_logits[mask].float() / temperature, dim=-1)
    with torch.no_grad():
        teacher = F.softmax(teacher_logits[mask].float() / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature ** 2


class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model, loss_lambda=0.5, temperature=1.5, **kwargs):
        if not 0 <= loss_lambda <= 1 or temperature <= 0:
            raise ValueError("Require 0 <= loss_lambda <= 1 and temperature > 0")
        super().__init__(*args, **kwargs)
        if self.args.n_gpu > 1:
            raise ValueError(
                "多卡蒸馏请使用 torchrun --nproc_per_node=<GPU数量> train_distil.py，"
                "不要使用单进程 DataParallel；每个进程需要独立的教师模型。")
        # Each distributed process owns a frozen teacher on its training device.
        self.teacher_model = teacher_model.to(self.args.device).eval()
        self.teacher_model.requires_grad_(False)
        self.loss_lambda = loss_lambda
        self.temperature = temperature
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        # Both seq2seq models shift labels internally. Do not apply the causal-LM
        # logits[:-1]/labels[1:] shift from the reference distillation script.
        outputs = model(**inputs, use_cache=False, return_dict=True)
        # Explicit student supervision, averaged over non-padding target tokens.
        student_loss = (
            F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.size(-1)),
                            labels.reshape(-1), ignore_index=-100)
            if labels.ne(-100).any() else outputs.logits.sum() * 0.0
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, use_cache=False, return_dict=True)
        kl_loss = distillation_loss(outputs.logits, teacher_outputs.logits, labels, self.temperature)
        loss = self.loss_lambda * student_loss + (1 - self.loss_lambda) * kl_loss
        return (loss, outputs) if return_outputs else loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-rank", "--local_rank", dest="local_rank", type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
        help="torchrun 进程的本地 GPU 编号，通常由 LOCAL_RANK 自动设置")
    add = functools.partial(add_arguments, argparser=parser)
    for name, default, help_text in [
        ("train_data", "dataset/train.json", "训练数据列表"),
        ("test_data", "dataset/test.json", "验证数据列表"),
        ("teacher_model", "openai/whisper-small", "Whisper 教师模型 ID 或完整模型目录"),
        ("teacher_adapter", None, "教师的可选 LoRA/AdaLoRA 适配器目录"),
        ("student_model", None, "Whisper/Distil-Whisper 模型 ID 或本地 Whisper/Transformer 目录；默认随机初始化 Transformer"),
        ("output_dir", "output/distillation", "输出目录"),
        ("language", "Chinese", "语言，None 表示多语言"),
        ("task", "transcribe", "transcribe 或 translate"),
        ("augment_config_path", None, "训练数据增强配置"),
        ("resume_from_checkpoint", None, "Trainer 检查点目录"),
    ]:
        add(name, type=str, default=default, help=help_text)
    for name, default in [
        ("warmup_steps", 50), ("logging_steps", 100), ("eval_steps", 1000),
        ("save_steps", 1000), ("save_total_limit", 3), ("num_workers", 8),
        ("per_device_train_batch_size", 4), ("per_device_eval_batch_size", 4),
        ("gradient_accumulation_steps", 1), ("student_d_model", 256),
        ("student_encoder_layers", 6), ("student_decoder_layers", 4),
        ("student_attention_heads", 4), ("student_ffn_dim", 1024), ("seed", 42),
    ]:
        add(name, type=int, default=default, help=name)
    for name, default in [
        ("learning_rate", 1e-4), ("num_train_epochs", 3.0),
        ("temperature", 1.5), ("min_audio_len", 0.5), ("max_audio_len", 30.0),
        ("student_dropout", 0.1), ("max_grad_norm", 1.0),
    ]:
        add(name, type=float, default=default, help=name)
    for name, default in [
        ("fp16", torch.cuda.is_available()), ("bf16", False), ("timestamps", False),
        ("local_files_only", False), ("use_compile", False),
    ]:
        add(name, type=bool, default=default, help=name)
    parser.add_argument(
        "--lambda", "--loss_lambda", "--alpha", dest="loss_lambda", type=float, default=0.5,
        help="学生 CE 权重：[0, 1]，KL 权重为 1-lambda。默认 0.5；--alpha 为兼容别名。")
    args = parser.parse_args(argv)
    if not 0 <= args.loss_lambda <= 1 or args.temperature <= 0:
        parser.error("lambda 必须在 [0, 1] 内，temperature 必须大于 0")
    if args.fp16 and args.bf16:
        parser.error("fp16 和 bf16 不能同时启用；使用 bf16 时设置 --fp16 False")
    if args.task not in ("transcribe", "translate"):
        parser.error("task 必须是 transcribe 或 translate")
    if not 0.5 <= args.min_audio_len <= args.max_audio_len <= 30:
        parser.error("要求 0.5 <= min_audio_len <= max_audio_len <= 30")
    if args.eval_steps <= 0 or args.save_steps <= 0 or args.save_steps % args.eval_steps:
        parser.error("save_steps 必须是 eval_steps 的正整数倍，以便加载最佳模型")
    if platform.system() == "Windows":
        args.num_workers = 0
    return args


def setup_distributed_device(args):
    """Bind this worker before loading models; Trainer initializes DDP itself."""
    # Environment takes priority, including when resuming with torchrun.
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank >= 0:
        os.environ["LOCAL_RANK"] = str(local_rank)
        if torch.cuda.is_available():
            if local_rank >= torch.cuda.device_count():
                raise ValueError(
                    f"LOCAL_RANK={local_rank} 超出可见 GPU 数量；"
                    "请检查 CUDA_VISIBLE_DEVICES 和 --nproc_per_node")
            torch.cuda.set_device(local_rank)
    args.local_rank = local_rank


def build_student(args, teacher_config, processor):
    """Load a compatible pretrained student, or build the original Transformer."""
    # Resume also restores architecture; CLI size defaults must not override it.
    checkpoint = args.resume_from_checkpoint or args.student_model
    if checkpoint:
        config = AutoConfig.from_pretrained(
            checkpoint, local_files_only=args.local_files_only)
        if config.model_type not in ("whisper", TransformerConfig.model_type):
            raise ValueError(
                f"不支持的学生架构：{config.model_type}；仅支持 Whisper/Distil-Whisper "
                f"和本地 {TransformerConfig.model_type} Transformer")
    else:
        config = TransformerConfig(
            vocab_size=teacher_config.vocab_size,
            num_mel_bins=teacher_config.num_mel_bins,
            d_model=args.student_d_model,
            encoder_layers=args.student_encoder_layers,
            decoder_layers=args.student_decoder_layers,
            encoder_attention_heads=args.student_attention_heads,
            decoder_attention_heads=args.student_attention_heads,
            encoder_ffn_dim=args.student_ffn_dim,
            decoder_ffn_dim=args.student_ffn_dim,
            dropout=args.student_dropout,
            max_source_positions=teacher_config.max_source_positions,
            max_target_positions=teacher_config.max_target_positions,
            pad_token_id=teacher_config.pad_token_id,
            bos_token_id=teacher_config.bos_token_id,
            eos_token_id=teacher_config.eos_token_id,
            decoder_start_token_id=teacher_config.decoder_start_token_id,
        )
    # Validate before allocating pretrained weights (potentially several GB).
    for name in ("vocab_size", "num_mel_bins", "pad_token_id", "bos_token_id",
                 "eos_token_id", "decoder_start_token_id"):
        if getattr(config, name) != getattr(teacher_config, name):
            raise ValueError(
                f"学生与教师的 {name} 不一致：学生={getattr(config, name)}，"
                f"教师={getattr(teacher_config, name)}。当前蒸馏要求共享词表和音频特征；"
                "例如 whisper-small 可搭配 whisper-large-v2，不能直接搭配 whisper-large-v3。")
    if max(processor.tokenizer.get_vocab().values()) >= config.vocab_size:
        raise ValueError("教师词表大小不足以容纳 processor 的 token ID")
    if processor.feature_extractor.feature_size != config.num_mel_bins:
        raise ValueError("音频特征维度与模型不一致")
    if checkpoint:
        student_processor = WhisperProcessor.from_pretrained(
            checkpoint, local_files_only=args.local_files_only)
        if student_processor.tokenizer.get_vocab() != processor.tokenizer.get_vocab():
            raise ValueError("学生检查点与教师的 token-ID 映射不一致")
        if student_processor.feature_extractor.feature_size != config.num_mel_bins:
            raise ValueError("学生 processor 的音频特征维度与模型不一致")
        student = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint, config=config, local_files_only=args.local_files_only)
    else:
        student = TransformerForConditionalGeneration(config)
    if config.model_type == "whisper":
        # Whisper's encoder sinusoidal positions are fixed. Some Transformers
        # loaders lose requires_grad=False; restore it for stable optimizer groups
        # across initialization and checkpoint resume.
        student.model.encoder.embed_positions.requires_grad_(False)
    return student


def main(argv=None):
    args = parse_args(argv)
    setup_distributed_device(args)
    is_main_process = int(os.environ.get("RANK", 0)) == 0
    if is_main_process:
        print_arguments(args)
    set_seed(args.seed)
    # Keep data/audio dependencies out of the CLI and loss-only imports.
    from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
    from utils.reader import CustomDataset

    processor = WhisperProcessor.from_pretrained(
        args.teacher_model, language=args.language, task=args.task,
        no_timestamps=not args.timestamps, local_files_only=args.local_files_only)
    teacher = WhisperForConditionalGeneration.from_pretrained(
        args.teacher_model, local_files_only=args.local_files_only)
    if args.teacher_adapter:
        from peft import PeftModel
        teacher = PeftModel.from_pretrained(
            teacher, args.teacher_adapter, is_trainable=False).merge_and_unload()
    teacher.requires_grad_(False).eval()
    teacher.config.use_cache = False
    student = build_student(args, teacher.config, processor)
    datasets = []
    for path, augment in [(args.train_data, args.augment_config_path), (args.test_data, None)]:
        if Path(path).is_dir():
            # All ranks must see the same sample order before sharding.
            path = [str(x) for x in sorted(Path(path).glob("*.json"))]
        dataset = CustomDataset(
            data_list_path=path, processor=processor, language=args.language,
            timestamps=args.timestamps, min_duration=args.min_audio_len,
            max_duration=args.max_audio_len, augment_config_path=augment)
        if not len(dataset):
            raise ValueError(f"数据集为空，请检查路径及过滤条件：{path}")
        datasets.append(dataset)
    if is_main_process:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch = (args.per_device_train_batch_size
                           * args.gradient_accumulation_steps * world_size)
        print(f"训练数据：{len(datasets[0])}，验证数据：{len(datasets[1])}")
        print(f"学生架构：{student.config.model_type}，"
              f"学生参数：{sum(p.numel() for p in student.parameters()):,}")
        print(f"训练进程数：{world_size}，有效全局 batch size：{effective_batch}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        local_rank=args.local_rank,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs, max_grad_norm=args.max_grad_norm,
        fp16=args.fp16, bf16=args.bf16, torch_compile=args.use_compile,
        eval_strategy="steps", save_strategy="steps",
        eval_steps=args.eval_steps, save_steps=args.save_steps,
        logging_steps=args.logging_steps, save_total_limit=args.save_total_limit,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, prediction_loss_only=True,
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=args.num_workers, optim="adamw_torch",
        ddp_find_unused_parameters=False, report_to="none", seed=args.seed,
    )
    trainer = DistillationTrainer(
        model=student, teacher_model=teacher, loss_lambda=args.loss_lambda,
        temperature=args.temperature, args=training_args,
        train_dataset=datasets[0], eval_dataset=datasets[1],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor, decoder_start_token_id=teacher.config.decoder_start_token_id),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    final_dir = os.path.join(args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
