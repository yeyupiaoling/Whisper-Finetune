import csv
import json
import os.path

import click
import soundfile
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("openai/whisper-base")


@click.command()
@click.option("--folder", "-f", required=True, type=str,
              help='Path to the dataset folder')
def prepare_dataset(folder):
    with open(os.path.join(folder, "metadata.csv"), 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    test_lines = []
    train_lines = []
    for row in rows:
        file_name = row[0]
        text = row[1]
        if len(tokenizer.encode(text)) > 448:
            continue
        line = {"audio": {"path": os.path.join(folder, file_name).replace("\\", "/")}, "sentence": text}
        if file_name.startswith("test/") or file_name.startswith("test\\"):
            test_lines.append(line)
        elif file_name.startswith("train/") or file_name.startswith("train\\"):
            train_lines.append(line)

    for i in tqdm(range(len(test_lines))):
        audio_path = test_lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        test_lines[i]["duration"] = duration
        test_lines[i]["sentences"] = [{"start": 0, "end": duration, "text": test_lines[i]["sentence"]}]

    f = open(os.path.join(folder, "test.json"), 'w', encoding='utf-8')
    for line in test_lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()

    for i in tqdm(range(len(train_lines))):
        audio_path = train_lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        train_lines[i]["duration"] = duration
        train_lines[i]["sentences"] = [{"start": 0, "end": duration, "text": train_lines[i]["sentence"]}]

    f = open(os.path.join(folder, "train.json"), 'w', encoding='utf-8')
    for line in train_lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()


if __name__ == '__main__':
    prepare_dataset()
