import argparse
import os
import shutil

from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration

from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lora_model", type=str, default="output/checkpoint-16803/adapter_model", help="微调保存的模型路径")
parser.add_argument('--output_dir', type=str, default='models/', help="合并模型的保存目录")
args = parser.parse_args()

# 检查模型文件是否存在
assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)
# 获取Whisper的基本模型
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
# 与Lora模型合并
model = PeftModel.from_pretrained(base_model, args.lora_model)

# 保存的文件夹路径
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)

# 把Whisper中的一些配置文件都复制过来
p = snapshot_download(repo_id=peft_config.base_model_name_or_path, local_files_only=True)
files = os.listdir(p)
for f in files:
    if f.endswith('.json') or f.endswith('.py') or f.endswith('.txt'):
        shutil.copyfile(os.path.join(p, f), os.path.join(save_directory, f))

# 合并参数
model = model.merge_and_unload()
model.train(False)

lora_model_sd = model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

# 保存模型到指定目录中
WhisperForConditionalGeneration.save_pretrained(base_model,
                                                save_directory=save_directory,
                                                state_dict=deloreanized_sd)
print(f'合并模型保持在：{save_directory}')
