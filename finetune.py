import argparse
import functools
import os

import torch
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, AdaLoraConfig
from peft import prepare_model_for_int8_training
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, get_audio_length_processor
from utils.utils import print_arguments, SavePeftModelCallback, make_inputs_require_grad, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="dataset/test.json",        help="测试数据集的路径")
add_arg("base_model",    type=str, default="openai/whisper-tiny",      help="Whisper的基础模型")
add_arg("output_dir",    type=str, default="output",                   help="训练保存模型的路径")
add_arg("warmup_steps",  type=int, default=50,      help="训练预热步数")
add_arg("logging_steps", type=int, default=100,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=10000,   help="多少步数评估一次")
add_arg("save_steps",    type=int, default=10000,   help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=8,       help="读取数据的线程数量")
add_arg("learning_rate", type=float,  default=1e-3, help="学习率大小")
add_arg("min_audio_len", type=float,  default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float,  default=30,   help="最大的音频长度，单位秒")
add_arg("use_adalora",   type=bool,   default=True, help="是否使用AdaLora而不是Lora")
add_arg("fp16",          type=bool,   default=True, help="是否使用fp16训练模型")
add_arg("use_8bit",      type=bool,   default=True, help="是否将模型量化为8位")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=3,    help="训练的轮数")
add_arg("language",      type=str, default="Chinese", help="设置语言")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=8,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=8,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("generation_max_length",       type=int, default=128,  help="训练数据的最大长度")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.base_model), f"模型文件{args.base_model}不存在，请检查是否为huggingface存在模型"
# 获取Whisper的特征提取器、编码器和解码器
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model, local_files_only=args.local_files_only)
tokenizer = WhisperTokenizer.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)


# 数据预处理
def prepare_dataset(batch):
    new_batch = {}
    # 从输入音频数组中计算log-Mel输入特征
    new_batch["input_features"] = [feature_extractor(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
                                   for a in batch["audio"]]
    # 将目标文本编码为标签ID
    new_batch["labels"] = [tokenizer(s).input_ids for s in batch["sentence"]]
    return new_batch


# 数据加载
audio_dataset = load_dataset('json', data_files={'train': args.train_data, 'test': args.test_data})
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
print(f"过滤前训练数据：{audio_dataset['train'].num_rows}，测试数据：{audio_dataset['test'].num_rows}")
# 过滤时长不在指定区域的音频
if 'duration' in audio_dataset['train'].features.keys():
    is_audio_in_length = get_audio_length_processor(args.min_audio_len, args.max_audio_len)
    audio_dataset["train"] = audio_dataset["train"].filter(is_audio_in_length, input_columns=["duration"])
    audio_dataset["test"] = audio_dataset["test"].filter(is_audio_in_length, input_columns=["duration"])
    print(f"过滤后训练数据：{audio_dataset['train'].num_rows}，测试数据：{audio_dataset['test'].num_rows}")
audio_dataset = audio_dataset.with_transform(prepare_dataset)
# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 获取Whisper模型
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    args.per_device_train_batch_size = args.per_device_train_batch_size * world_size

if args.use_8bit:
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                            load_in_8bit=True,
                                                            device_map=device_map,
                                                            local_files_only=args.local_files_only)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # 转化为Lora模型
    model = prepare_model_for_int8_training(model)
else:
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                            device_map=device_map,
                                                            local_files_only=args.local_files_only)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
# 设置Lora参数
if args.use_adalora:
    config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                           lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5,
                           target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"])
else:
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
# 恢复训练时加载Lora参数
if args.resume_from_checkpoint:
    adapters_dict = torch.load(f'{args.resume_from_checkpoint}/pytorch_model.bin')
    set_peft_model_state_dict(model=model, peft_model_state_dict=adapters_dict)

# 定义训练参数
training_args = Seq2SeqTrainingArguments(output_dir=args.output_dir,
                                         per_device_train_batch_size=args.per_device_train_batch_size,
                                         gradient_accumulation_steps=args.gradient_accumulation_steps,
                                         learning_rate=args.learning_rate,
                                         warmup_steps=args.warmup_steps,
                                         num_train_epochs=args.num_train_epochs,
                                         save_strategy="steps",
                                         evaluation_strategy="steps",
                                         fp16=args.fp16,
                                         report_to=["tensorboard"],
                                         save_steps=args.save_steps,
                                         eval_steps=args.eval_steps,
                                         save_total_limit=5,
                                         optim='adamw_torch',
                                         load_best_model_at_end=True,
                                         ddp_find_unused_parameters=False if ddp else None,
                                         dataloader_num_workers=args.num_workers,
                                         per_device_eval_batch_size=args.per_device_eval_batch_size,
                                         generation_max_length=args.generation_max_length,
                                         logging_steps=args.logging_steps,
                                         remove_unused_columns=False,
                                         label_names=["labels"])

if training_args.local_rank == 0 or training_args.local_rank == -1:
    print('=' * 90)
    model.print_trainable_parameters()
    print('=' * 90)

# 定义训练器
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=audio_dataset["train"],
                         eval_dataset=audio_dataset["test"],
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SavePeftModelCallback])
model.config.use_cache = False

if training_args.local_rank == 0 or training_args.local_rank == -1:
    print("如果加载恢复训练参数，出现miss keys警告，请忽略它。")
# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存最后的模型
trainer.save_state()
if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))
