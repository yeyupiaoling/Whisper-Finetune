import argparse
import json
import os

import ijson
from tqdm import tqdm
import sys

sys.path.insert(0, sys.path[0] + "/../")
from utils.binary import DatasetWriter


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wenetspeech_json', type=str, default='/media/WenetSpeech数据集/WenetSpeech.json',
                    help="WenetSpeech的标注json文件路径")
parser.add_argument('--pun_model_path', type=str, default=None,
                    help="添加标点符号的模型，模型来源：https://github.com/yeyupiaoling/PunctuationModel")
parser.add_argument('--annotation_dir', type=str, default='dataset/', help="存放数据列表的文件夹路径")
args = parser.parse_args()

# 使用符号模型
if args.pun_model_path:
    from utils.pun_predictor import PunctuationExecutor
    pun_executor = PunctuationExecutor(model_dir=args.pun_model_path, use_gpu=True)

if not os.path.exists(args.annotation_dir):
    os.makedirs(args.annotation_dir)
# 训练数据列表
train_list_path = os.path.join(args.annotation_dir, 'train.json')
f_train = open(train_list_path, 'w', encoding='utf-8')
# 测试数据列表
test_net_path = os.path.join(args.annotation_dir, 'test_net.json')
test_meeting_path = os.path.join(args.annotation_dir, 'test_meeting.json')
f_test_net = open(test_net_path, 'w', encoding='utf-8')
f_test_meeting = open(test_meeting_path, 'w', encoding='utf-8')


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
    all_data = get_data(args.wenetspeech_json)
    print(f'总数据量为：{len(all_data)}')
    for data in tqdm(all_data):
        long_audio_path, segments_lists = data
        for segment_file in segments_lists:
            start_time = float(segment_file['begin_time'])
            end_time = float(segment_file['end_time'])
            text = segment_file['text']
            # 添加标点符号
            if args.pun_model_path:
                text = pun_executor(text)
            confidence = segment_file['confidence']
            if confidence < 0.95: continue
            line = dict(audio={"path": long_audio_path,
                               "start_time": round(start_time, 3),
                               "end_time": round(end_time, 3)},
                        sentence=text,
                        duration=round(end_time - start_time, 3))
            data_type = long_audio_path.split('/')[-4]
            if data_type == 'test_net':
                f_test_net.write('{}\n'.format(str(line).replace("'", '"')))
            if data_type == 'test_meeting':
                f_test_meeting.write('{}\n'.format(str(line).replace("'", '"')))
            if data_type == 'train':
                f_train.write('{}\n'.format(str(line).replace("'", '"')))
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
                # 对文本最后如果是感叹号，就转成句号，因为很多感叹号标注不准确
                if sentence[-1] == '！':
                    if sentence[-2] == '啊' or sentence[-2] == '呀':
                        text += sentence
                    else:
                        sentence = sentence[:-1] + '。'
                        text += sentence
                else:
                    text += sentence
                name = data['audio']['path']
                if i < len(lines) - 2:
                    next_data = json.loads(lines[i + 1])
                    next_name = next_data['audio']['path']
                    # 如果下一条数据是新数据或者加上就大于30秒，就写入数据
                    if next_name != name or duration + next_data['duration'] >= 30:
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
    # 转成二进制文件，减少内存占用
    create_binary()

