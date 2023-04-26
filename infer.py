import argparse

import soundfile
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor

from utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", type=str, default="dataset/test.wav",              help="预测的音频路径")
parser.add_argument("--language",   type=int, default="Chinese",                       help="设置语言")
parser.add_argument("--model_path", type=str, default="models/whisper-large-v2-finetune",  help="合并模型的路径，或者是huggingface上模型的名称")
parser.add_argument("--task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
args = parser.parse_args()
print_arguments(args)

# 获取Whisper的特征提取器、编码器和解码器
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)
processor = WhisperProcessor.from_pretrained(args.model_path, language=args.language, task=args.task)

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path, device_map="auto")
model.eval()

# 读取音频
sample, sr = soundfile.read(args.audio_path)
# 预处理音频
input_features = processor(sample, sampling_rate=sr, return_tensors="pt").input_features.cuda()
# 开始识别
predicted_ids = model.generate(input_features)
# 解码结果
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"识别结果：{transcription}")
