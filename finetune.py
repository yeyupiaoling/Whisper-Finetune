import argparse
import functools
import os
import platform

from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="dataset/test.json",        help="测试数据集的路径")
add_arg("base_model",    type=str, default="openai/whisper-tiny",      help="Whisper的基础模型")
add_arg("output_dir",    type=str, default="output/",                  help="训练保存模型的路径")
add_arg("warmup_steps",  type=int, default=50,      help="训练预热步数")
add_arg("logging_steps", type=int, default=100,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=1000,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=1000,    help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=8,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=1e-3,  help="学习率大小")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒，不能大于30秒")
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
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=8,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=8,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("push_to_hub",                 type=bool, default=False, help="是否将模型权重推到HuggingFace Hub")
add_arg("hub_model_id",                type=str,  default=None,  help="HuggingFace Hub上的模型仓库ID")
add_arg("save_total_limit",            type=int,  default=10,  help="只保存最新检查点的数量")
args = parser.parse_args()
print_arguments(args)

# 如果是Windows，num_workers设置为0
if platform.system() == "Windows":
    args.num_workers = 0


def main():
    # 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
    processor = WhisperProcessor.from_pretrained(args.base_model,
                                                 language=args.language,
                                                 task=args.task,
                                                 no_timestamps=not args.timestamps,
                                                 local_files_only=args.local_files_only)

    # 读取数据
    train_dataset = CustomDataset(data_list_path=args.train_data,
                                  processor=processor,
                                  language=args.language,
                                  timestamps=args.timestamps,
                                  min_duration=args.min_audio_len,
                                  max_duration=args.max_audio_len,
                                  augment_config_path=args.augment_config_path)
    test_dataset = CustomDataset(data_list_path=args.test_data,
                                 processor=processor,
                                 language=args.language,
                                 timestamps=args.timestamps,
                                 min_duration=args.min_audio_len,
                                 max_duration=args.max_audio_len)
    print(f"训练数据：{len(train_dataset)}，测试数据：{len(test_dataset)}")
    # 数据padding器
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 获取Whisper模型
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # 获取模型
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                            load_in_8bit=args.use_8bit,
                                                            device_map=device_map,
                                                            local_files_only=args.local_files_only)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # 量化模型
    model = prepare_model_for_kbit_training(model)
    # 注册forward，否则多卡训练会失败
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    print('加载LoRA模块...')
    if args.resume_from_checkpoint:
        # 恢复训练时加载Lora参数
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        print(f'adding LoRA modules...')
        target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        print(target_modules)
        if args.use_adalora:
            total_step = args.num_train_epochs * len(train_dataset)
            config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                   lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules,
                                   total_step=total_step)
        else:
            config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)

    if args.base_model.endswith("/"):
        args.base_model = args.base_model[:-1]
    output_dir = str(os.path.join(args.output_dir, os.path.basename(args.base_model)))
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

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print('=' * 90)
        model.print_trainable_parameters()
        print('=' * 90)

    # 定义训练器
    trainer = Seq2SeqTrainer(args=training_args,
                             model=model,
                             train_dataset=train_dataset,
                             eval_dataset=test_dataset,
                             data_collator=data_collator,
                             processing_class=processor.feature_extractor,
                             callbacks=[SavePeftModelCallback])
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # 开始训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存最后的模型
    trainer.save_state()
    # 重新启用缓存以更快地推断
    model.config.use_cache = True
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    # 是否把模型参数文件推送到huggingface
    if training_args.push_to_hub:
        hub_model_id = args.hub_model_id if args.hub_model_id is not None else output_dir
        model.push_to_hub(hub_model_id)


if __name__ == '__main__':
    main()
