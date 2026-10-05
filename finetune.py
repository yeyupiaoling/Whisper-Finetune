import argparse
import functools
import os
import platform

import torch

from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)
from transformers.trainer_utils import get_last_checkpoint

from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.reader import CustomDataset, DistillWhisperDataset, MoonshineDataset
from utils.utils import print_arguments, add_arguments


DATASET_CLASSES = {
    "whisper": CustomDataset,
    "distill_whisper": DistillWhisperDataset,
    "moonshine": MoonshineDataset,
}
DATASET_ALIASES = {cls.__name__: name for name, cls in DATASET_CLASSES.items()}

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="训练数据集的JSON/JSONL文件或目录路径（目录仅扫描当前层）")
add_arg("test_data",     type=str, default="dataset/test.json",        help="测试数据集的JSON/JSONL文件或目录路径（目录仅扫描当前层）")
add_arg("dataset_type", type=str, default="auto",
        choices=["auto", *DATASET_CLASSES, *DATASET_ALIASES],
    help="数据集类型或 reader.py 中的类名；auto 根据 base_model 名称选择")
add_arg("base_model", type=str, default="openai/whisper-tiny", help="基础模型的 Hugging Face ID 或本地目录；名称需包含 whisper、distil 或 moonshine")
add_arg("output_dir",    type=str, default="output/",                  help="训练保存模型的路径")
add_arg("warmup_steps",  type=int, default=50,      help="训练预热步数")
add_arg("logging_steps", type=int, default=100,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=1000,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=1000,    help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=8,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=1e-5,  help="学习率大小")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒，不能大于30秒")
add_arg("lora_enable",   type=bool, default=False, help="是否使用LoRA/AdaLoRA；False为全参数微调")
add_arg("use_adalora",   type=bool,  default=True,  help="是否使用AdaLora而不是Lora")
add_arg("fp16",          type=bool,  default=True,  help="是否使用fp16训练模型")
add_arg("use_8bit",      type=bool,  default=False, help="是否将模型量化为8位")
add_arg("timestamps",    type=bool,  default=False, help="训练时是否使用时间戳数据")
add_arg("use_compile",   type=bool, default=False, help="是否使用Pytorch2.0的编译器")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=3,      help="训练的轮数")
add_arg("language", type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则训练的是多语言")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("augment_config_path",         type=str, default=None, help="数据增强配置文件路径")
add_arg("resume_from_checkpoint",      type=str, default="auto", help="默认auto：从output_dir下当前模型的最新断点恢复；none：从头训练；也可指定断点路径")
add_arg("per_device_train_batch_size", type=int, default=8,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=8,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("push_to_hub",                 type=bool, default=False, help="是否将模型权重推到HuggingFace Hub")
add_arg("hub_model_id",                type=str,  default=None,  help="HuggingFace Hub上的模型仓库ID")
add_arg("save_total_limit",            type=int,  default=10,  help="只保存最新检查点的数量")


def parse_args(argv=None):
    args = parser.parse_args(argv)
    # model_type仅作为内部推断结果，不再需要命令行传入。
    model_name = (args.base_model or "").lower()
    if "moonshine" in model_name:
        args.model_type = "moonshine"
    elif "distil" in model_name:
        args.model_type = "distill_whisper"
    elif "whisper" in model_name:
        args.model_type = "whisper"
    else:
        parser.error("无法从 base_model 名称识别模型，请使用包含 whisper、distil 或 moonshine 的名称或路径")
    if args.use_8bit and not args.lora_enable:
        parser.error("use_8bit=True 需要同时设置 lora_enable=True；全参数微调请关闭8位量化")
    args.dataset_type = DATASET_ALIASES.get(args.dataset_type, args.dataset_type)
    if args.dataset_type == "auto":
        args.dataset_type = args.model_type
    # Whisper 与 Distill-Whisper 的特征格式相同，Moonshine 则使用原始波形。
    if (args.model_type == "moonshine") != (args.dataset_type == "moonshine"):
        parser.error("Moonshine 模型必须搭配 MoonshineDataset；Whisper 模型不能使用该数据集")
    if args.model_type == "moonshine" and (args.timestamps or args.task != "transcribe"):
        parser.error("Moonshine 不支持 timestamps=True 或 task=translate")
    if platform.system() == "Windows":
        args.num_workers = 0
    return args


def load_processor(args):
    if args.model_type == "moonshine":
        processor = AutoProcessor.from_pretrained(
            args.base_model, local_files_only=args.local_files_only)
        processor.tokenizer.pad_token = "</s>"
        processor.tokenizer.eos_token = "</s>"
        processor.tokenizer.bos_token = "<s>"
        return processor
    # Distill-Whisper 也使用 Whisper tokenizer 和 log-Mel 特征。
    return WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task=args.task,
        no_timestamps=not args.timestamps, local_files_only=args.local_files_only)


def resolve_resume_checkpoint(args, output_dir):
    checkpoint = args.resume_from_checkpoint
    if checkpoint is None or checkpoint.lower() == "none":
        return None
    if checkpoint.lower() == "auto":
        # 仅匹配 checkpoint-数字，避免选到用于推理的 final 或 best。
        checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        if checkpoint is None:
            print(f"未找到训练断点，从头训练：{output_dir}")
            return None
    print(f"从断点恢复训练：{checkpoint}")
    return checkpoint


def main(argv=None):
    args = parse_args(argv)
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model.rstrip("/")))
    # 在加载 LoRA 前解析为实际路径，适配器和 Trainer 使用同一个断点。
    args.resume_from_checkpoint = resolve_resume_checkpoint(args, output_dir)
    print_arguments(args)
    processor = load_processor(args)

    # 读取数据
    dataset_class = DATASET_CLASSES[args.dataset_type]
    train_dataset = dataset_class(data_list_path=args.train_data,
                                  processor=processor,
                                  language=args.language,
                                  timestamps=args.timestamps,
                                  min_duration=args.min_audio_len,
                                  max_duration=args.max_audio_len,
                                  augment_config_path=args.augment_config_path)
    test_dataset = dataset_class(data_list_path=args.test_data,
                                 processor=processor,
                                 language=args.language,
                                 timestamps=args.timestamps,
                                 min_duration=args.min_audio_len,
                                 max_duration=args.max_audio_len)
    print(f"训练数据：{len(train_dataset)}，测试数据：{len(test_dataset)}")
    if not len(train_dataset) or not len(test_dataset):
        raise ValueError("训练集或测试集为空，请检查路径和音频时长过滤条件")

    # 配置模型加载设备
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # AutoModel 根据配置加载 Distill-Whisper / Moonshine，不依赖模型路径命名。
    model_class = (WhisperForConditionalGeneration if args.model_type == "whisper"
                   else AutoModelForSpeechSeq2Seq)
    if args.model_type == "distill_whisper":
        model_kwargs = dict(device_map=device_map, dtype=torch.float32, local_files_only=args.local_files_only)
    else:
        model_kwargs = dict(device_map=device_map, local_files_only=args.local_files_only)
    if args.use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = model_class.from_pretrained(args.base_model, **model_kwargs)
    expected_architecture = "moonshine" if args.model_type == "moonshine" else "whisper"
    if model.config.model_type != expected_architecture:
        raise ValueError(f"从 base_model 推断的类型 {args.model_type} 与模型架构 {model.config.model_type} 不匹配")
    if args.model_type != "moonshine":
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
    else:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    # 未启用梯度检查点，不需要依赖 Whisper encoder.conv1 的输入梯度 hook。
    if args.use_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if args.lora_enable:
        print('加载LoRA模块...')
        if args.resume_from_checkpoint:
            # LoRA恢复需要适配器检查点；全参数恢复由Trainer处理。
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        else:
            print('adding LoRA modules...')
            module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
            target_modules = [name for name in
                              ["k_proj", "q_proj", "v_proj", "out_proj", "o_proj", "fc1", "fc2"]
                              if name in module_names]
            if not target_modules:
                raise ValueError("模型中未找到支持的 LoRA 目标层")
            print(target_modules)
            if args.use_adalora:
                total_step = args.num_train_epochs * len(train_dataset)
                config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                       lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules,
                                       total_step=total_step)
            else:
                config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
            model = get_peft_model(model, config)

    # 定义训练参数
    training_args = \
        Seq2SeqTrainingArguments(output_dir=output_dir,  # 保存检查点和意志的目录
                                 per_device_train_batch_size=args.per_device_train_batch_size,  # 训练batch_size大小
                                 per_device_eval_batch_size=args.per_device_eval_batch_size,  # 评估batch_size大小
                                 gradient_accumulation_steps=args.gradient_accumulation_steps,  # 训练梯度累计步数
                                 learning_rate=args.learning_rate,  # 学习率大小
                                 warmup_steps=args.warmup_steps,  # 预热步数
                                 num_train_epochs=args.num_train_epochs,  # 微调训练轮数
                                 save_strategy="steps",  # 指定按照步数保存检查点
                                 eval_strategy="steps",  # 指定按照步数评估模型
                                 load_best_model_at_end=True,  # 指定是否在结束时加载最优模型
                                 fp16=args.fp16,  # 是否使用半精度训练
                                 report_to=["tensorboard"],  # 指定使用tensorboard保存log
                                 save_steps=args.save_steps,  # 指定保存检查点的步数
                                 eval_steps=args.eval_steps,  # 指定评估模型的步数
                                 torch_compile=args.use_compile,  # 使用Pytorch2.0的编译器
                                 save_total_limit=args.save_total_limit,  # 只保存最新检查点的数量
                                 optim='adamw_torch',  # 指定优化方法
                                 ddp_find_unused_parameters=False if ddp else None,  # 分布式训练设置
                                 dataloader_num_workers=args.num_workers,  # 设置读取数据的线程数量
                                 logging_steps=args.logging_steps,  # 指定打印log的步数
                                 remove_unused_columns=False,  # 删除模型不需要的数据列
                                 label_names=["labels"],  # 与标签对应的输入字典中的键列表
                                 push_to_hub=args.push_to_hub, # 是否将模型权重推到HuggingFace Hub
                                 )

    if args.lora_enable and training_args.local_rank in (0, -1):
        print('=' * 90)
        model.print_trainable_parameters()
        print('=' * 90)

    # 定义训练器
    trainer = Seq2SeqTrainer(args=training_args,
                             model=model,
                             train_dataset=train_dataset,
                             eval_dataset=test_dataset,
                             data_collator=data_collator,
                             processing_class=processor,
                             callbacks=[SavePeftModelCallback] if args.lora_enable else [])
    model.config.use_cache = False

    # 使用Trainer原生恢复逻辑，避免空的load_from_checkpoint跳过权重恢复。
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存最后的模型
    trainer.save_state()
    # 重新启用缓存以更快地推断
    model.config.use_cache = True
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
        processor.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    # 是否把模型参数文件推送到huggingface
    if training_args.push_to_hub:
        hub_model_id = args.hub_model_id if args.hub_model_id is not None else output_dir
        model.push_to_hub(hub_model_id)


if __name__ == '__main__':
    main()
