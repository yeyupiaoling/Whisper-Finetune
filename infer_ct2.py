import argparse
import json
import os

from faster_whisper import WhisperModel

from utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--audio_path",  type=str,  default="dataset/test.wav",        help="预测的音频路径")
parser.add_argument("--language",    type=int, default="Chinese",                  help="设置语言")
parser.add_argument("--model_path",  type=str,  default="models/whisper-large-v2-ct2", help="转换后的模型路径，转换方式看文档")
parser.add_argument("--use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
parser.add_argument("--use_int8",    type=bool, default=False,  help="是否使用int8进行预测")
parser.add_argument("--beam_size",   type=int,  default=10,     help="解码搜索大小")
parser.add_argument("--num_workers", type=int,  default=1,      help="预测器的并发数量")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.model_path), f"模型文件{args.model_path}不存在"
# 加载模型
if args.use_gpu:
    if not args.use_int8:
        model = WhisperModel(args.model_path, device="cuda", compute_type="float16", num_workers=args.num_workers)
    else:
        model = WhisperModel(args.model_path, device="cuda", compute_type="int8_float16", num_workers=args.num_workers)
else:
    model = WhisperModel(args.model_path, device="cpu", compute_type="int8", num_workers=args.num_workers)
# 预热
_, _ = model.transcribe("dataset/test.wav", beam_size=5)


# 语音识别
def run_recognize(path):
    segments, info = model.transcribe(path, beam_size=args.beam_size, language=args.language)
    results = []
    result_text = ''
    for segment in segments:
        text = segment.text
        result_text += text
        results.append(dict(start=round(segment.start, 2), end=round(segment.end, 2), text=text))
    result = dict(language=info.language, duration=info.duration, results=results, text=result_text)
    return result


if __name__ == '__main__':
    recognize_result = run_recognize(args.audio_path)
    print("识别结果：")
    # 打印结果
    print(json.dumps(recognize_result, ensure_ascii=False, indent=4))
