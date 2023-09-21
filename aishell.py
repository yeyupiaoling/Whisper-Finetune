import argparse
import json
import os
import functools

import soundfile
from tqdm import tqdm

from utils.utils import download, unpack
from utils.utils import add_arguments, print_arguments

DATA_URL = 'https://openslr.elda.org/resources/33/data_aishell.tgz'
MD5_DATA = '2f494334227864a8a8fec932999db9d8'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("filepath", default=None, type=str, help="压缩包data_aishell.tgz文件路径，不指定会自动下载")
add_arg("target_dir", default="dataset/audio/", type=str, help="存放音频文件的目录")
add_arg("annotation_text", default="dataset/", type=str, help="存放音频标注文件的目录")
add_arg('add_pun', default=False, type=bool, help="是否添加标点符")
args = parser.parse_args()


def create_annotation_text(data_dir, annotation_path):
    print('Create Aishell annotation text ...')
    if args.add_pun:
        import logging
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        from modelscope.utils.logger import get_logger
        logger = get_logger(log_level=logging.CRITICAL)
        logger.setLevel(logging.CRITICAL)
        inference_pipline = pipeline(task=Tasks.punctuation,
                                     model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
                                     model_revision="v1.0.0")
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    f_train = open(os.path.join(annotation_path, 'train.json'), 'w', encoding='utf-8')
    f_test = open(os.path.join(annotation_path, 'test.json'), 'w', encoding='utf-8')
    transcript_path = os.path.join(data_dir, 'transcript', 'aishell_transcript_v0.8.txt')
    transcript_dict = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        if line == '': continue
        audio_id, text = line.split(' ', 1)
        # remove space
        text = ''.join(text.split())
        if args.add_pun:
            text = inference_pipline(text_in=text)['text']
        transcript_dict[audio_id] = text
    # 训练集
    data_types = ['train', 'dev']
    lines = []
    for type in data_types:
        audio_dir = os.path.join(data_dir, 'wav', type)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname)
                audio_id = fname[:-4]
                # if no transcription for audio then skipped
                if audio_id not in transcript_dict:
                    continue
                text = transcript_dict[audio_id]
                line = {"audio": {"path": audio_path}, "sentence": text}
                lines.append(line)
    # 添加音频时长
    for i in tqdm(range(len(lines))):
        audio_path = lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        lines[i]["duration"] = duration
        lines[i]["sentences"] = [{"start": 0, "end": duration, "text": lines[i]["sentence"]}]
    for line in lines:
        f_train.write(json.dumps(line, ensure_ascii=False) + "\n")
    # 测试集
    audio_dir = os.path.join(data_dir, 'wav', 'test')
    lines = []
    for subfolder, _, filelist in sorted(os.walk(audio_dir)):
        for fname in filelist:
            audio_path = os.path.join(subfolder, fname)
            audio_id = fname[:-4]
            # if no transcription for audio then skipped
            if audio_id not in transcript_dict:
                continue
            text = transcript_dict[audio_id]
            line = {"audio": {"path": audio_path}, "sentence": text}
            lines.append(line)
    # 添加音频时长
    for i in tqdm(range(len(lines))):
        audio_path = lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        lines[i]["duration"] = duration
        lines[i]["sentences"] = [{"start": 0, "end": duration, "text": lines[i]["sentence"]}]
    for line in lines:
        f_test.write(json.dumps(line,  ensure_ascii=False)+"\n")
    f_test.close()
    f_train.close()


def prepare_dataset(url, md5sum, target_dir, annotation_path, filepath=None):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_aishell')
    if not os.path.exists(data_dir):
        if filepath is None:
            filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        # unpack all audio tar files
        audio_dir = os.path.join(data_dir, 'wav')
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for ftar in filelist:
                unpack(os.path.join(subfolder, ftar), subfolder, True)
        os.remove(filepath)
    else:
        print("Skip downloading and unpacking. Aishell data already exists in %s." % target_dir)
    create_annotation_text(data_dir, annotation_path)


def main():
    print_arguments(args)
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(url=DATA_URL,
                    md5sum=MD5_DATA,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text,
                    filepath=args.filepath)


if __name__ == '__main__':
    main()
