# 将ljspeech格式数据转换为jsonlines
# ljspeech数据格式为 LJ001-0002|in being comparatively modern.|in being comparatively modern.
import json
import os.path
import random
import click
import soundfile
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("openai/whisper-base")


def deal_rows(rows, folder, output_file):
    lines = []
    for row in rows:
        if "|" not in row:
            continue
        file_name, text = row.split("|")
        if len(tokenizer.encode(text)) > 448:
            continue
        line = {"audio": {"path": os.path.join(folder, "wavs", file_name.replace(".wav", "") + ".wav")},
                "sentence": text}
        lines.append(line)
    for i in tqdm(range(len(lines))):
        audio_path = lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        lines[i]["duration"] = duration
        lines[i]["sentences"] = [{"start": 0, "end": duration, "text": lines[i]["sentence"]}]
    f = open(output_file, 'w', encoding='utf-8')
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()


@click.command()
@click.option("--folder", "-f", required=True, type=str,
              help='Path to the dataset folder')
def prepare_dataset(folder):
    rows = open(os.path.join(folder, "metadata.csv"), 'r', encoding='utf-8').read().split("\n")
    random.shuffle(rows)
    split_index = len(rows) // 10
    test_rows = rows[:split_index]
    train_rows = rows[split_index:]
    deal_rows(train_rows, folder, os.path.join("dataset", "train.json"))
    deal_rows(test_rows, folder, os.path.join("dataset", "test.json"))


if __name__ == '__main__':
    prepare_dataset()
