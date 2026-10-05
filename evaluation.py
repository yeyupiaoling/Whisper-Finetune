import argparse
import functools
import json
import os
import platform
import math

from typing import List

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, MoonshineForConditionalGeneration
from transformers import WhisperProcessor, AutoProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation
from utils.reader import CustomDataset, MoonshineDataset
from utils.utils import print_arguments, add_arguments
from utils.text_processor import TextProcess

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.json",            help="测试集的JSON/JSONL文件或目录路径（目录仅扫描当前层）")
add_arg("model_path",  type=str, default="models/whisper-large-v3-finetune", help="待评估的完整模型或LoRA检查点路径，也可以是HuggingFace模型名称")
add_arg("base_model",  type=str, default=None, help="训练使用的基础模型；检查点缺少tokenizer或评估LoRA时需要指定")
add_arg("lora_enable", type=bool, default=False, help="是否加载未合并的LoRA/AdaLoRA检查点")
add_arg("max_new_tokens", type=int, default=255, help="生成文本的最大token数量")
add_arg("batch_size",  type=int, default=8,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("metric",     type=str, default="wer",        choices=['cer', 'wer'],              help="评估方式")
add_arg("result_path", type=str, default="evaluation_results.jsonl", help="预测文本与参考文本的JSONL保存路径，每次评估覆盖写入")


class EvaluationDataCollator(DataCollatorSpeechSeq2SeqWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "input_values" in batch:
            # Moonshine使用变长波形，不能将补零部分当作有效音频。
            lengths = torch.tensor([len(item["input_values"][0]) for item in features])
            width = batch["input_values"].shape[1]
            positions = torch.arange(width).unsqueeze(0)
            if self.processor.feature_extractor.padding_side == "left":
                mask = positions >= (width - lengths).unsqueeze(1)
            else:
                mask = positions < lengths.unsqueeze(1)
            batch["attention_mask"] = mask.long()
        return batch
    

def get_float_value_from_range(range: List[float], sa: float) -> float:
    """beamRangeとSAからbeam幅を決める。

    Args:
        range (List[float]): rangeパラメータ
        sa (float): SA
        return_int (bool): Trueならintを返す。

    Returns:
        float: saに対応する値
    """
    assert len(range) >= 1
    assert 0.0 <= sa <= 1.0
    if len(range) >= 2:
        delta      = 1.0 / (len(range) - 1)
        lowerX     = math.floor((len(range) - 1) * sa)
        if lowerX == len(range) - 1:
            value = range[lowerX]
        else:
            diff = (range[lowerX + 1] - range[lowerX]) / delta * (sa - lowerX * delta)
            value = int(range[lowerX] + diff)
    else:
        value = range[0]
    return value


def main(args):
    if platform.system() == "Windows":
        args.num_workers = 0

    if args.lora_enable and not args.base_model:
        raise ValueError("评估未合并的LoRA检查点时，请用 --base_model 指定训练的基础模型")

    # 权重始终来自待评估的检查点；训练只保存feature_extractor时，
    # 通过base_model加载完整processor，而不是误加载基础模型权重。
    model_source = args.base_model if args.lora_enable else args.model_path
    processor_source = args.base_model or args.model_path
    config = AutoConfig.from_pretrained(model_source, local_files_only=args.local_files_only)
    
    if config.model_type not in ("whisper", "moonshine"):
        raise ValueError(f"不支持的模型类型：{config.model_type}")
    is_moonshine = config.model_type == "moonshine"
    if is_moonshine and args.timestamps:
        raise ValueError("Moonshine不支持Whisper格式的时间戳标签，请设置 --timestamps False")

    try:
        if is_moonshine:
            processor = AutoProcessor.from_pretrained(
                processor_source, local_files_only=args.local_files_only)
            processor.tokenizer.pad_token = "</s>"
            processor.tokenizer.eos_token = "</s>"
            processor.tokenizer.bos_token = "<s>"
        else:
            # Distil-Whisper沿用Whisper的processor和数据格式。
            processor = WhisperProcessor.from_pretrained(
                processor_source, language=args.language, task=args.task,
                no_timestamps=not args.timestamps,
                local_files_only=args.local_files_only)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"无法从 {processor_source} 加载完整processor。若检查点未保存tokenizer，"
            "请用 --base_model 指定训练时的基础模型，并检查 --local_files_only 设置。"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if is_moonshine:
        model_class = MoonshineForConditionalGeneration
    elif "distil" in processor_source.lower() or "distil" in model_source.lower():
        model_class = AutoModelForSpeechSeq2Seq
    else:
        model_class = WhisperForConditionalGeneration
    model = model_class.from_pretrained(
        model_source, dtype=torch_dtype, local_files_only=args.local_files_only)
    if args.lora_enable:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, args.model_path, local_files_only=args.local_files_only)
        model = model.merge_and_unload()
    model.to(device)
    model.config.use_cache = True

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens, 
        "return_dict_in_generate": False
    }

    beam_width = int(get_float_value_from_range([1, 8, 20], 0.5))
    model.generation_config.use_cache = True
    model.generation_config.num_beams = beam_width
    model.generation_config.num_return_sequences = 1
    model.generation_config.length_penalty = 0.7

    if not is_moonshine:
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        generation_kwargs["return_timestamps"] = args.timestamps
        # 英语专用Whisper/Distil-Whisper不接受language/task参数。
        if getattr(model.generation_config, "is_multilingual", False):
            generation_kwargs["language"] = args.language.lower() if args.language else None
            generation_kwargs["task"] = args.task
        else:
            model.generation_config.language = None
            model.generation_config.task = None
    model.eval()

    dataset_class = MoonshineDataset if is_moonshine else CustomDataset
    test_dataset = dataset_class(
        data_list_path=args.test_data, processor=processor, language=args.language,
        timestamps=args.timestamps, min_duration=args.min_audio_len,
        max_duration=args.max_audio_len)
    print(f"测试数据：{len(test_dataset)}")
    if len(test_dataset) == 0:
        raise ValueError("测试数据为空，请检查数据路径和音频长度过滤条件")
    eval_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=EvaluationDataCollator(processor=processor))
    metric_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", f"{args.metric}.py")
    metric = evaluate.load(metric_path)
    
    text_process  = TextProcess(
        normalize=True, case="lower", half_full="half", remove_marks=True
    )
    input_name = "input_values" if is_moonshine else "input_features"
    os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as result_file:
        for batch in tqdm(eval_dataloader):
            inputs = {input_name: batch[input_name].to(device=device, dtype=model.dtype)}
            if "attention_mask" in batch:
                inputs["attention_mask"] = batch["attention_mask"].to(device)
            # 不从真实标签截取decoder_input_ids，避免将答案泄露给生成过程。
            with torch.inference_mode():
                generated_tokens = model.generate(**inputs, **generation_kwargs).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            for i in range(len(decoded_labels)):
                decoded_preds[i] = text_process(decoded_preds[i])
                decoded_labels[i] = text_process(decoded_labels[i])
                result_file.write(json.dumps({
                    "decoded_preds": decoded_preds[i],
                    "decoded_labels": decoded_labels[i],
                }, ensure_ascii=False) + "\n")
            result_file.flush()
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    print(f"预测与参考文本已保存：{args.result_path}")
    m = metric.compute()
    print(f"评估结果：{args.metric}={round(m, 5)}")


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    main(args)
