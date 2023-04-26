import argparse
import os

import torch
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft import prepare_model_for_int8_training
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.utils import print_arguments, SavePeftModelCallback

parser = argparse.ArgumentParser()
parser.add_argument("--train_data",    type=str, default="dataset/train.json",       help="训练数据集的路径")
parser.add_argument("--test_data",     type=str, default="dataset/test.json",        help="测试数据集的路径")
parser.add_argument("--base_model",    type=str, default="openai/whisper-tiny",      help="Whisper的基础模型")
parser.add_argument("--output_path",   type=str, default="models/whisper-tiny-lora", help="训练保存模型的路径")
parser.add_argument("--warmup_steps",  type=int, default=50,      help="训练预热步数")
parser.add_argument("--logging_steps", type=int, default=100,     help="打印日志步数")
parser.add_argument("--eval_steps",    type=int, default=10000,   help="多少步数评估一次")
parser.add_argument("--save_steps",    type=int, default=10000,   help="多少步数保存模型一次")
parser.add_argument("--num_workers",   type=int, default=8,       help="读取数据的线程数量")
parser.add_argument("--learning_rate", type=float,  default=1e-3, help="学习率大小")
parser.add_argument("--use_8bit",      type=bool,   default=False, help="是否将模型量化为8位")
parser.add_argument("--num_train_epochs", type=int, default=3,    help="训练的轮数")
parser.add_argument("--language",      type=str, default="Chinese", help="设置语言")
parser.add_argument("--task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
parser.add_argument("--resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
parser.add_argument("--per_device_train_batch_size", type=int, default=8,    help="训练的batch size")
parser.add_argument("--per_device_eval_batch_size",  type=int, default=8,    help="评估的batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
parser.add_argument("--generation_max_length",       type=int, default=128,  help="训练数据的最大长度")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.base_model), f"模型文件{args.base_model}不存在，请检查是否为huggingface存在模型"
# 获取Whisper的特征提取器、编码器和解码器
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
tokenizer = WhisperTokenizer.from_pretrained(args.base_model, language=args.language, task=args.task)
processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)


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
audio_data = load_dataset('json', data_files={'train': args.train_data, 'test': args.test_data})
audio_data = audio_data.cast_column("audio", Audio(sampling_rate=16000))
audio_data = audio_data.with_transform(prepare_dataset)
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
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, load_in_8bit=True, device_map=device_map)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # 转化为Lora模型
    model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
else:
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, device_map=device_map).half()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
# 恢复训练时加载Lora参数
if args.resume_from_checkpoint:
    adapters_dict = torch.load(f'{args.resume_from_checkpoint}/pytorch_model.bin')
    set_peft_model_state_dict(model=model, peft_model_state_dict=adapters_dict)
# 打印训练参数
print("="*70)
model.print_trainable_parameters()
print("="*70)

# 定义训练参数
training_args = Seq2SeqTrainingArguments(output_dir="output",
                                         per_device_train_batch_size=args.per_device_train_batch_size,
                                         gradient_accumulation_steps=args.gradient_accumulation_steps,
                                         learning_rate=args.learning_rate,
                                         warmup_steps=args.warmup_steps,
                                         num_train_epochs=args.num_train_epochs,
                                         save_strategy="steps",
                                         evaluation_strategy="steps",
                                         fp16=True,
                                         report_to=["tensorboard"],
                                         save_steps=args.save_steps,
                                         eval_steps=args.eval_steps,
                                         save_total_limit=5,
                                         load_best_model_at_end=True,
                                         dataloader_num_workers=args.num_workers,
                                         per_device_eval_batch_size=args.per_device_eval_batch_size,
                                         generation_max_length=args.generation_max_length,
                                         logging_steps=args.logging_steps,
                                         remove_unused_columns=False,
                                         label_names=["labels"])

# 定义训练器
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=audio_data["train"],
                         eval_dataset=audio_data["test"],
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SavePeftModelCallback])
model.config.use_cache = False

print("如果加载恢复训练参数，出现miss keys警告，请忽略它。")
# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存最后的模型
if training_args.local_rank == 0:
    model.save_pretrained(training_args.output_dir)
