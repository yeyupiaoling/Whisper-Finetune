import argparse
import functools
import platform

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM

from utils.utils import print_arguments, add_arguments

import speech_recognition as sr
import wave
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="dataset/test.wav", help="预测的音频路径")
add_arg("model_path",  type=str,  default="models/tiny-finetune/", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("use_gpu",     type=bool, default=True,      help="是否使用gpu进行预测")
add_arg("language",    type=str,  default="chinese", help="设置语言，如果为None则预测的是多语言")
add_arg("num_beams",   type=int,  default=1,         help="解码搜索大小")
add_arg("batch_size",  type=int,  default=16,        help="预测batch_size大小")
add_arg("use_compile", type=bool, default=False,     help="是否使用Pytorch2.0的编译器")
add_arg("task",        type=str,  default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("assistant_model_path",  type=str,  default=None,  help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
add_arg("local_files_only",      type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
add_arg("use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
args = parser.parse_args()
print_arguments(args)


def save_as_wav(audio, output_file_path):
    with wave.open(output_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位PCM编码
        wav_file.setframerate(44100)  # 采样率为44.1kHz
        wav_file.writeframes(audio.frame_data)


def input_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说...")
        r.pause_threshold = 1
        audio = r.listen(source)
        args.audio_path = "dataset/temp_file.wav"
        save_as_wav(audio, args.audio_path)


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

# 推理参数
generate_kwargs = {"task": args.task, "num_beams": args.num_beams}
if args.language is not None:
    generate_kwargs["language"] = args.language

if args.audio_path is False:
    print('-----------audio_path is False---------------')
    input_audio()

    result = infer_pipe(args.audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
    os.remove(args.audio_path)
else:
    # 推理
    result = infer_pipe(args.audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)

results = ''
for chunk in result["chunks"]:
    results = results + chunk['text']
    print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")

print('---------------')
print(results)
