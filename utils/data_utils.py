import re
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch


# 删除标点符号
def remove_punctuation(text: str or List[str]):
    punctuation = '!,.;:?、！，。；：？'
    if isinstance(text, str):
        text = re.sub(r'[{}]+'.format(punctuation), '', text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r'[{}]+'.format(punctuation), '', t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_name = self.processor.feature_extractor.model_input_names[0]
        input_features = [{input_name: feature[input_name][0]} for feature in features]
        # Moonshine 使用变长原始波形，需要 mask 区分真实音频与 padding。
        padding_kwargs = {"return_attention_mask": True} if input_name == "input_values" else {}
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", **padding_kwargs)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        start_token_id = self.decoder_start_token_id
        if start_token_id is None:
            start_token_id = self.processor.tokenizer.bos_token_id
        if labels.shape[1] and start_token_id is not None and (labels[:, 0] == start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
