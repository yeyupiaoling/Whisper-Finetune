# Copyright 2021 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Character Error Ratio (CER) metric."""

from typing import List

import datasets
from jiwer import cer
import jiwer.transforms as tr
from datasets.config import PY_VERSION
from packaging import version
import evaluate


if PY_VERSION < version.parse("3.8"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


SENTENCE_DELIMITER = ""


if version.parse(importlib_metadata.version("jiwer")) < version.parse("2.3.0"):

    class SentencesToListOfCharacters(tr.AbstractTransform):
        def __init__(self, sentence_delimiter: str = " "):
            self.sentence_delimiter = sentence_delimiter

        def process_string(self, s: str):
            return list(s)

        def process_list(self, inp: List[str]):
            chars = []
            for sent_idx, sentence in enumerate(inp):
                chars.extend(self.process_string(sentence))
                if (
                    self.sentence_delimiter is not None
                    and self.sentence_delimiter != ""
                    and sent_idx < len(inp) - 1
                ):
                    chars.append(self.sentence_delimiter)
            return chars

    cer_transform = tr.Compose(
        [
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            SentencesToListOfCharacters(SENTENCE_DELIMITER),
        ]
    )
else:
    cer_transform = tr.Compose(
        [
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
            tr.ReduceToListOfListOfChars(),
        ]
    )


_CITATION = """\
@inproceedings{inproceedings,
    author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
    year = {2004},
    month = {01},
    pages = {},
    title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
"""

_DESCRIPTION = """\
Character error rate (CER) is a common metric of the performance of an automatic speech recognition system.

CER is similar to Word Error Rate (WER), but operates on character instead of word. Please refer to docs of WER for further information.

Character error rate can be computed as:

CER = (S + D + I) / N = (S + D + I) / (S + D + C)

where

S is the number of substitutions,
D is the number of deletions,
I is the number of insertions,
C is the number of correct characters,
N is the number of characters in the reference (N=S+D+C).

CER's output is not always a number between 0 and 1, in particular when there is a high number of insertions. This value is often associated to the percentage of characters that were incorrectly predicted. The lower the value, the better the
performance of the ASR system with a CER of 0 being a perfect score.
"""

_KWARGS_DESCRIPTION = """
Computes CER score of transcribed segments against references.
Args:
    references: list of references for each speech input.
    predictions: list of transcribtions to score.
    concatenate_texts: Whether or not to concatenate sentences before evaluation, set to True for more accurate result.
Returns:
    (float): the character error rate

Examples:

    >>> predictions = ["this is the prediction", "there is an other sample"]
    >>> references = ["this is the reference", "there is another one"]
    >>> cer = evaluate.load("cer")
    >>> cer_score = cer.compute(predictions=predictions, references=references)
    >>> print(cer_score)
    0.34146341463414637
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CER(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
                "https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates",
            ],
        )

    # 新版
    def _compute(self, predictions, references, concatenate_texts=False):
        # 新版本：cer_transform 说明要用字符级错误率，所以用cer()函数
        if concatenate_texts:
            # 连接所有文本成单个字符串（字符级处理）
            ref_str = " ".join(references)
            hyp_str = " ".join(predictions)
            return cer(
                ref_str,
                hyp_str,
                reference_transform=cer_transform,
                hypothesis_transform=cer_transform,
            )

        # 重点改造：不再需要measures，直接用cer计算+累积
        total_edit_distance = 0  # 总编辑距离（错误字符数）
        total_ref_chars = 0  # 总参考字符数

        for reference, prediction in zip(references, predictions):
            # 关键：用cer()直接计算当前句子的错误率
            error_rate = cer(
                reference,
                prediction,
                reference_transform=cer_transform,
                hypothesis_transform=cer_transform,
            )

            # 重要：通过错误率 * 参考字符数 = 编辑距离
            # （因为 CER = 编辑距离 / 参考字符数）
            edit_distance = error_rate * len(reference)
            total_edit_distance += edit_distance
            total_ref_chars += len(reference)

        # 保护除零错误（新版本必须加）
        return total_edit_distance / total_ref_chars if total_ref_chars else 0.0
