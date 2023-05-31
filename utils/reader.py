import json
import os
from json import JSONDecodeError
from typing import List

import librosa
import soundfile
from torch.utils.data import Dataset

from utils.binary import DatasetReader


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 no_timestamps=True,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30):
        super(CustomDataset, self).__init__()
        self.data_list_path = data_list_path
        self.processor = processor
        self.data_list_path = data_list_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.no_timestamps = no_timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()

    # 加载数据列表
    def _load_data_list(self):
        if self.data_list_path.endswith(".header"):
            # 获取二进制的数据列表
            self.dataset_reader = DatasetReader(data_header_path=self.data_list_path,
                                                min_duration=self.min_duration,
                                                max_duration=self.max_duration)
            self.data_list = self.dataset_reader.get_keys()
        else:
            # 获取数据列表
            with open(self.data_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in lines:
                if isinstance(line, str):
                    line = json.loads(line)
                if not isinstance(line, dict): continue
                # 跳过超出长度限制的音频
                if line["duration"] < self.min_duration:
                    continue
                if self.max_duration != -1 and line["duration"] > self.max_duration:
                    continue
                self.data_list.append(dict(line))

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentence"] if self.no_timestamps else data_list["sentences"]
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            # 分割读取音频
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        if self.mono:
            sample = librosa.to_mono(sample)
        if self.sample_rate != sample_rate:
            sample = librosa.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
        return sample, sample_rate, transcript

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['decoder_input_ids'] = labels
        data['labels'] = labels[1:] + [self.vocab['<|endoftext|>']]
        return data

    def __getitem__(self, idx):
        # 从数据列表里面获取音频数据、采样率和文本
        sample, sample_rate, transcript = self._get_list_data(idx=idx)
        if self.no_timestamps:
            # 获取log-Mel特征和标签ID
            data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
        else:
            # 加载带有时间戳的文本
            data = self._load_timestamps_transcript(transcript=transcript)
            # 从输入音频数组中计算log-Mel输入特征
            data["input_features"] = self.processor(audio=sample, sampling_rate=self.sample_rate).input_features
        return data

    def __len__(self):
        return len(self.data_list)

    # 分割读取音频
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate
