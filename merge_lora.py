import argparse
import functools
import os

from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast,\
    WhisperProcessor
from peft import PeftModel, PeftConfig
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("lora_model", type=str, default="output/whisper-tiny/checkpoint-best/", help="微调保存的模型路径")
add_arg('output_dir', type=str, default='models/',    help="合并模型的保存目录")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)
# 获取Whisper的基本模型
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map={"": "cpu"},
                                                             local_files_only=args.local_files_only)
# 与Lora模型合并
model = PeftModel.from_pretrained(base_model, args.lora_model, local_files_only=args.local_files_only)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path,
                                                            local_files_only=args.local_files_only)
tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                 local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                             local_files_only=args.local_files_only)

# 合并参数
model = model.merge_and_unload()
model.train(False)

# 保存的文件夹路径
if peft_config.base_model_name_or_path.endswith("/"):
    peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)

# 保存模型到指定目录中
model.save_pretrained(save_directory, max_shard_size='4GB')
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
