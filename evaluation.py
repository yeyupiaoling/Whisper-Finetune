import argparse
import functools
import gc
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, get_audio_length_processor, remove_punctuation, \
    to_simple
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.json",            help="测试集的路径")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("metric",     type=str, default="cer",        choices=['cer', 'wer'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"
# 获取Whisper的特征提取器、编码器和解码器
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only)
model.eval()


# 数据预处理
def prepare_dataset(batch):
    new_batch = {}
    # 从输入音频数组中计算log-Mel输入特征
    new_batch["input_features"] = [feature_extractor(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
                                   for a in batch["audio"]]
    # 将目标文本编码为标签ID
    new_batch["labels"] = [processor.tokenizer(s).input_ids for s in batch["sentence"]]
    return new_batch


# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 加载评估数据
audio_dataset = load_dataset('json', data_files={'test': args.test_data})
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
print(f"过滤前测试数据：{audio_dataset['test'].num_rows}")
# 过滤时长不在指定区域的音频
if 'duration' in audio_dataset['test'].features.keys():
    is_audio_in_length = get_audio_length_processor(args.min_audio_len, args.max_audio_len)
    audio_dataset["test"] = audio_dataset["test"].filter(is_audio_in_length, input_columns=["duration"])
    print(f"过滤后测试数据：{audio_dataset['test'].num_rows}")
audio_dataset = audio_dataset.with_transform(prepare_dataset)
eval_dataloader = DataLoader(audio_dataset['test'], batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# 获取评估方法
metric = evaluate.load(args.metric)

# 开始评估
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :4].cuda(),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255).cpu().numpy())
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # 将预测和实际的token转换为文本
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # 删除标点符号
            if args.remove_pun:
                decoded_preds = remove_punctuation(decoded_preds)
                decoded_labels = remove_punctuation(decoded_labels)
            # 将繁体中文总成简体中文
            if args.to_simple:
                decoded_preds = to_simple(decoded_preds)
                decoded_labels = to_simple(decoded_labels)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    # 删除计算的记录
    del generated_tokens, labels, batch
    gc.collect()
# 计算评估结果
m = metric.compute()
print(f"评估结果：{args.metric}={round(m, 5)}")
