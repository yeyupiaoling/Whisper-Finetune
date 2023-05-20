import argparse
import functools

import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path", type=str, default="dataset/test.wav",              help="预测的音频路径")
add_arg("model_path", type=str, default="models/whisper-tiny-finetune",  help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("language",   type=str, default="Chinese",                       help="设置语言，可全称也可简写，如果为None则预测的是多语言")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("local_files_only", type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 获取Whisper的特征提取器、编码器和解码器
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only).half()
model.eval()

# 读取音频
sample, sr = librosa.load(args.audio_path, sr=16000)
duration = sample.shape[-1]/sr
assert duration < 30, f"本程序只适合推理小于30秒的音频，当前音频{duration}秒，请使用其他推理程序!"
# 预处理音频
input_features = processor(sample, sampling_rate=sr, return_tensors="pt", do_normalize=True).input_features.cuda().half()
# 开始识别
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256)
# 解码结果
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"识别结果：{transcription}")
