import argparse
import gc
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--test_data",   type=str, default="dataset/test.json",            help="测试集的路径")
parser.add_argument("--model_path",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
parser.add_argument("--batch_size",  type=int, default=16,        help="评估的batch size")
parser.add_argument("--num_workers", type=int, default=8,         help="读取数据的线程数量")
parser.add_argument("--language",    type=str, default="Chinese", help="设置语言")
parser.add_argument("--task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
parser.add_argument("--metric",     type=str, default="cer",        choices=['cer', 'wer'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"
# 获取Whisper的特征提取器、编码器和解码器
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)
tokenizer = WhisperTokenizer.from_pretrained(args.model_path, language=args.language, task=args.task)
processor = WhisperProcessor.from_pretrained(args.model_path, language=args.language, task=args.task)


# 数据预处理
def prepare_dataset(batch):
    new_batch = {}
    # 从输入音频数组中计算log-Mel输入特征
    new_batch["input_features"] = [feature_extractor(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
                                   for a in batch["audio"]]
    # 将目标文本编码为标签ID
    new_batch["labels"] = [tokenizer(s).input_ids for s in batch["sentence"]]
    return new_batch


# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 加载评估数据
audio_data = load_dataset('json', data_files={'test': args.test_data})
audio_data = audio_data.cast_column("audio", Audio(sampling_rate=16000))
audio_data = audio_data.with_transform(prepare_dataset)
eval_dataloader = DataLoader(audio_data['test'], batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# 获取评估方法
metric = evaluate.load(args.metric)

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path, device_map="auto")
model.eval()

# 开始评估
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :4].cuda(),
                    max_new_tokens=255).cpu().numpy())
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # 将预测和实际的token转换为文本
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    # 删除计算的记录
    del generated_tokens, labels, batch
    gc.collect()
# 计算评估结果
m = metric.compute()
print(f"评估结果：{args.metric}={round(m, 5)}")