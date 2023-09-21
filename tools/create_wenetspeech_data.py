import argparse
import json
import logging
import math
import multiprocessing
import os
import sys
from multiprocessing import cpu_count

import ijson
import soundfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../")
from utils.binary import DatasetWriter

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wenetspeech_json', type=str, default='/media/WenetSpeech数据集/WenetSpeech.json',
                    help="WenetSpeech的标注json文件路径")
parser.add_argument('--add_pun', type=bool, default=True, help="是否添加标点符")
parser.add_argument('--annotation_dir', type=str, default='../dataset/', help="存放数据列表的文件夹路径")
args = parser.parse_args()

if not os.path.exists(args.annotation_dir):
    os.makedirs(args.annotation_dir)

# 训练、测试数据列表
train_list_path = os.path.join(args.annotation_dir, 'train_wenet.json')
test_net_path = os.path.join(args.annotation_dir, 'test_net.json')
test_meeting_path = os.path.join(args.annotation_dir, 'test_meeting.json')


# 获取标注信息
def get_data(wenetspeech_json):
    data_list = []
    input_dir = os.path.dirname(wenetspeech_json)
    i = 0
    # 开始读取数据，因为文件太大，无法获取进度
    with open(wenetspeech_json, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'audios.item')
        print("开始读取数据")
        while True:
            try:
                long_audio = objects.__next__()
                i += 1
                try:
                    long_audio_path = os.path.realpath(os.path.join(input_dir, long_audio['path']))
                    aid = long_audio['aid']
                    segments_lists = long_audio['segments']
                    assert (os.path.exists(long_audio_path))
                except AssertionError:
                    print(f'''Warning: {long_audio_path} 不存在或者已经处理过自动删除了，跳过''')
                    continue
                except Exception:
                    print(f'''Warning: {aid} 数据读取错误，跳过''')
                    continue
                else:
                    data_list.append([long_audio_path.replace('\\', '/'), segments_lists])
            except StopIteration:
                print("数据读取完成")
                break
    return data_list


def main():
    f_train = open(train_list_path, 'w', encoding='utf-8')
    f_test_net = open(test_net_path, 'w', encoding='utf-8')
    f_test_meeting = open(test_meeting_path, 'w', encoding='utf-8')

    all_data = get_data(args.wenetspeech_json)
    print(f'总数据量为：{len(all_data)}')
    for data in tqdm(all_data):
        long_audio_path, segments_lists = data
        for segment_file in segments_lists:
            start_time = float(segment_file['begin_time'])
            end_time = float(segment_file['end_time'])
            text = segment_file['text']
            confidence = segment_file['confidence']
            if confidence < 0.95: continue
            line = dict(audio={"path": long_audio_path,
                               "start_time": round(start_time, 3),
                               "end_time": round(end_time, 3)},
                        sentence=text,
                        duration=round(end_time - start_time, 3))
            data_type = long_audio_path.split('/')[-4]
            if data_type == 'test_net':
                f_test_net.write(json.dumps(line, ensure_ascii=False) + '\n')
            if data_type == 'test_meeting':
                f_test_meeting.write(json.dumps(line, ensure_ascii=False) + '\n')
            if data_type == 'train':
                f_train.write(json.dumps(line, ensure_ascii=False) + '\n')
    f_train.close()
    f_test_meeting.close()
    f_test_net.close()


# 合并多条音频，增加时间戳，同时加速训练
def merge_list():
    for file_path in [train_list_path, test_net_path, test_meeting_path]:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(file_path, 'w', encoding='utf-8') as f:
            sentences = []
            duration = 0
            start_time = 0
            text = ''
            for i in tqdm(range(len(lines))):
                data = json.loads(lines[i])
                sentence = data["sentence"]
                # 新数据
                if duration == 0:
                    start_time = data['audio']["start_time"]
                duration = data['audio']["end_time"] - start_time
                # 带时间戳数据
                sentences.append({"start": round(data['audio']["start_time"] - start_time, 2),
                                  "end": round(data['audio']['end_time'] - start_time, 2),
                                  "text": sentence})
                text += sentence
                name = data['audio']['path']
                if i < len(lines) - 2:
                    next_data = json.loads(lines[i + 1])
                    next_name = next_data['audio']['path']
                    next_end_time = next_data['audio']["end_time"]
                    # 如果下一条数据是新数据或者加上就大于30秒，就写入数据
                    if next_name != name or next_end_time - start_time >= 30:
                        data1 = dict()
                        data1['audio'] = {"path": data['audio']['path']}
                        data1['audio']['start_time'] = start_time
                        data1['audio']['end_time'] = data['audio']['end_time']
                        data1['duration'] = round(data['audio']['end_time'] - start_time, 2)
                        data1['sentence'] = text
                        data1['sentences'] = sentences
                        f.write(f'{json.dumps(data1, ensure_ascii=False)}\n')
                        sentences = []
                        duration = 0
                        start_time = 0
                        text = ''
                else:
                    # 最后一条数据处理方式
                    data1 = dict()
                    data1['audio'] = {"path": data['audio']['path']}
                    data1['audio']['start_time'] = start_time
                    data1['audio']['end_time'] = data['audio']['end_time']
                    data1['duration'] = round(data['audio']['end_time'] - start_time, 2)
                    data1['sentence'] = text
                    data1['sentences'] = sentences
                    f.write(f'{json.dumps(data1, ensure_ascii=False)}\n')
                    sentences = []
                    duration = 0
                    start_time = 0
                    text = ''


# 设置空白音频和转换格式
def process_audio(data, i):
    for path, sentences in tqdm(data, desc=f"处理进程{i}"):
        if not os.path.exists(path): continue
        save_path = path[:-5] + '.flac'
        if os.path.exists(save_path): continue
        sample, sr = soundfile.read(path)
        for sentence in sentences:
            start, end = sentence
            start = max(int((start + 0.1) * sr), 0)
            end = min(int((end - 0.1) * sr), len(sample))
            sample[start:end] = 0
        soundfile.write(save_path, sample, sr)


# 设置没有标注的位置静音
def set_silence():
    for file_path in [train_list_path, test_net_path]:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        all_data = {}
        for line in tqdm(lines, desc='读取数据列表'):
            data = json.loads(line)
            path = data['audio']['path']
            if os.path.splitext(path)[-1] != '.opus': continue
            start_a = data['audio']['start_time']
            sentences = data['sentences']
            last_end = start_a
            for sentence in sentences:
                start = round(start_a + sentence['start'], 3)
                if start - last_end > 1:
                    if path in all_data.keys():
                        all_data[path].append([last_end, start])
                    else:
                        all_data[path] = [[last_end, start]]
                else:
                    if path not in all_data.keys():
                        all_data[path] = []
                last_end = round(start_a + sentence['end'], 3)
        # 多进程处理数据
        all_data = list(all_data.items())
        num_worker = cpu_count()
        length = math.ceil(len(all_data) / num_worker)
        data = [all_data[i * length:(i + 1) * length] for i in range(num_worker)]
        my_process = []
        for i in range(num_worker):
            process = multiprocessing.Process(target=process_audio, args=(data[i], i))
            my_process.append(process)
        for process in my_process:
            process.start()
        for process in my_process:
            process.join()
        # 修改路径，因为是转成flac了
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in tqdm(lines, desc='修改路径后缀'):
                data = json.loads(line)
                path = data['audio']['path']
                path = path.replace('.opus', '.flac')
                if not os.path.exists(path):
                    print(f'{path}文件不存在', file=sys.stderr)
                    continue
                data['audio']['path'] = path
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


# 添加标点符号
def process_pun(data, i):
    inference_pipline = pipeline(task=Tasks.punctuation,
                                 model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
                                 model_revision="v1.0.0")
    f = open(f'temp{i}.txt', 'w', encoding='utf-8')
    for line in tqdm(data, desc=f"处理进程{i}"):
        data = json.loads(line)
        sentence = data['sentence']
        sentence = sentence.replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('、', '')
        sentence = inference_pipline(text_in=sentence)['text']
        data['sentence'] = sentence

        param_dict = {"cache": []}
        sentences = data['sentences']
        for i in range(len(sentences)):
            text = sentences[i]['text']
            text = text.replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('、', '')
            text = inference_pipline(text_in=text, param_dict=param_dict)['text']
            sentences[i]['text'] = text
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


# 多进程添加标点符号
def add_pun():
    for file_path in [train_list_path, test_net_path, test_meeting_path]:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_data = f.readlines()
        # 多进程添加标点符号，根据自己的显存大小调整
        num_worker = 4
        length = math.ceil(len(all_data) / num_worker)
        data = [all_data[i * length:(i + 1) * length] for i in range(num_worker)]
        my_process = []
        for i in range(num_worker):
            process = multiprocessing.Process(target=process_pun, args=(data[i], i))
            my_process.append(process)
        for process in my_process:
            process.start()
        for process in my_process:
            process.join()
        # 合并文件
        with open(file_path, 'w', encoding='utf-8') as fw:
            for i in range(num_worker):
                with open(f'temp{i}.txt', 'r', encoding='utf-8') as fr:
                    lines = fr.readlines()
                for line in lines:
                    fw.write(line)


# 转成二进制文件，减少内存占用
def create_binary():
    print('正在把数据列表转成二进制文件...')
    dataset_writer = DatasetWriter(f"{args.annotation_dir}/train")
    with open(train_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.replace('\n', '')
        dataset_writer.add_data(line)
    dataset_writer.close()


if __name__ == '__main__':
    main()
    # 合并多条音频，增加时间戳，同时加速训练
    merge_list()
    # 设置没有标注的位置静音
    set_silence()
    # 添加标点符号
    if args.add_pun:
        add_pun()
    # 转成二进制文件，减少内存占用
    create_binary()
