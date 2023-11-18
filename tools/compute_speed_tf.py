import argparse
import functools
import platform
import sys
import time

import soundfile
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM

sys.path.insert(0, sys.path[0] + "/../")
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="../dataset/test_long.wav", help="预测的音频路径")
add_arg("model_path",  type=str,  default="../base-model/openai/whisper-tiny", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("use_gpu",     type=bool, default=True,      help="是否使用gpu进行预测")
add_arg("num_beams",   type=int,  default=1,         help="解码搜索大小")
add_arg("infer_num",   type=int,  default=10,        help="预测的次数，不包括预热")
add_arg("batch_size",  type=int,  default=16,        help="预测batch_size大小")
add_arg("use_compile", type=bool, default=False,     help="是否使用Pytorch2.0的编译器")
add_arg("assistant_model_path",  type=str,  default=None,  help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
add_arg("local_files_only",      type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
add_arg("use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
args = parser.parse_args()
print_arguments(args)

# 设置设备
device = "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() and args.use_gpu else torch.float32

# 获取Whisper的特征提取器、编码器和解码器
processor = AutoProcessor.from_pretrained(args.model_path)

# 获取模型
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    use_flash_attention_2=args.use_flash_attention_2
)
if args.use_bettertransformer and not args.use_flash_attention_2:
    model = model.to_bettertransformer()
# 使用Pytorch2.0的编译器
if args.use_compile:
    if torch.__version__ >= "2" and platform.system().lower() != 'windows':
        model = torch.compile(model)
model.to(device)

# 获取助手模型
generate_kwargs_pipeline = None
if args.assistant_model_path is not None:
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    assistant_model.to(device)
    generate_kwargs_pipeline = {"assistant_model": assistant_model}

# 获取管道
infer_pipe = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      max_new_tokens=128,
                      chunk_length_s=30,
                      batch_size=args.batch_size,
                      torch_dtype=torch_dtype,
                      generate_kwargs=generate_kwargs_pipeline,
                      device=device)

sample, sr = soundfile.read(args.audio_path)
# 预热
_ = infer_pipe(sample.copy())

start_time = time.time()
for i in tqdm(range(args.infer_num)):
    _ = infer_pipe(sample.copy(), generate_kwargs={"task": "transcribe", "num_beams": args.num_beams})
print(f"音频时长：{int(len(sample) / sr)}s，预测平均耗时：{((time.time() - start_time) / args.infer_num):.3f}s")
