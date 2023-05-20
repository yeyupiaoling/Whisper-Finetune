import argparse
import os
import shutil
import threading

import ijson
from pydub import AudioSegment
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wenetspeech_json', type=str, default='/media/WenetSpeech数据集/WenetSpeech.json',
                    help="WenetSpeech的标注json文件路径")
parser.add_argument('--annotation_dir', type=str, default='dataset/', help="存放数据列表的文件夹路径")
args = parser.parse_args()

if not os.path.exists(args.annotation_dir):
    os.makedirs(args.annotation_dir)
# 训练数据列表
f_train = open(os.path.join(args.annotation_dir, 'train.json'), 'w', encoding='utf-8')
# 测试数据列表
f_test_net = open(os.path.join(args.annotation_dir, 'test_net.json'), 'w', encoding='utf-8')
f_test_meeting = open(os.path.join(args.annotation_dir, 'test_meeting.json'), 'w', encoding='utf-8')


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


if __name__ == '__main__':
    main()
