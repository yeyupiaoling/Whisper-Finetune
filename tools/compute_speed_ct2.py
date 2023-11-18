import argparse
import functools
import os
import sys
import time

import soundfile
from faster_whisper import WhisperModel
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../")
from utils.utils import print_arguments, add_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="../dataset/test_long.wav",    help="预测的音频路径")
add_arg("model_path",  type=str,  default="../models/whisper-tiny-ct2",  help="转换后的模型路径，转换方式看文档")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("infer_num",   type=int,  default=10,     help="预测的次数，不包括预热")
add_arg("use_int8",    type=bool, default=False,  help="是否使用int8进行预测")
add_arg("beam_size",   type=int,  default=1,      help="解码搜索大小")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.model_path), f"模型文件{args.model_path}不存在"
# 加载模型
if args.use_gpu:
    if not args.use_int8:
        model = WhisperModel(args.model_path, device="cuda", compute_type="float16",
                             local_files_only=args.local_files_only)
    else:
        model = WhisperModel(args.model_path, device="cuda", compute_type="int8_float16",
                             local_files_only=args.local_files_only)
else:
    model = WhisperModel(args.model_path, device="cpu", compute_type="int8",
                         local_files_only=args.local_files_only)
# 支持large-v3模型
if 'large-v3' in args.model_path:
    model.feature_extractor.mel_filters = \
        model.feature_extractor.get_mel_filters(model.feature_extractor.sampling_rate,
                                                model.feature_extractor.n_fft, n_mels=128)

sample, sr = soundfile.read(args.audio_path)
# 预热
_, _ = model.transcribe(sample.copy())

start_time = time.time()
# 语音识别
for i in tqdm(range(args.infer_num)):
    segments, info = model.transcribe(sample.copy(), beam_size=args.beam_size)
    for segment in segments:
        _ = segment.text
print(f"音频时长：{int(len(sample) / sr)}s，预测平均耗时：{((time.time() - start_time) / args.infer_num):.3f}s")
