import argparse
import json
import os
import re
from pathlib import Path

import soundfile


TRS_PATTERN = re.compile(
    r'\S+\s+(?P<audio_path>"[^"]+"|\S+)\s+"(?P<sentence>.*)"'
    r'(?:\s+-from\s+(?P<start>\S+)\s+-to\s+(?P<end>\S+))?'
)


def parse_trs_line(line: str, language=None, audio_root=None):
    """解析带切片时间或使用整段音频的 TRS，保留原始转写文本。"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # TRS 不是 shell 语法，不对撇号、反斜杠和文本内引号做转义处理。
    match = TRS_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError(f"不支持的 TRS 格式: {line}")

    # Docomo 数据的音频路径带双引号，其他数据通常不带。
    audio_path = match["audio_path"].strip('"')
    if audio_root is not None:
        audio_path = os.path.join(audio_root, audio_path)

    audio = {"path": audio_path}
    if match["start"] is not None:
        start_time = float(match["start"])
        end_time = float(match["end"])
        audio.update(start_time=start_time, end_time=end_time)
        duration = end_time - start_time
    else:
        # Dialect 等数据不带时间，CustomDataset 会读取整个文件。
        # 只读取音频元数据，不解码整段波形。
        duration = soundfile.info(audio_path).duration

    data = {
        "audio": audio,
        "sentence": match["sentence"],
        "duration": duration,
    }
    if language is not None:
        data["language"] = language
    return data


def convert_trs_files(
    trs_files,
    output_jsonl,
    language=None,
    audio_root=None,
    check_audio=False,
    append=False,
):
    """转换给定 TRS 文件；错误直接抛出，不捕获或静默跳过。"""
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with open(output_jsonl, "a" if append else "w", encoding="utf-8") as fout:
        for trs_file in trs_files:
            print(f"处理: {trs_file}")
            with open(trs_file, "r", encoding="utf-8-sig") as fin:
                for line_number, line in enumerate(fin, start=1):
                    data = parse_trs_line(line, language, audio_root)
                    if data is None:
                        continue

                    if check_audio and not os.path.isfile(data["audio"]["path"]):
                        raise FileNotFoundError(
                            f"{trs_file}:{line_number}: {data['audio']['path']}"
                        )

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    success += 1

    print("=" * 60)
    print(f"TRS文件数    : {len(trs_files)}")
    print(f"输出JSONL    : {output_jsonl}")
    print(f"成功         : {success}")
    print("=" * 60)


def convert_trs_file(
    trs_file,
    output_jsonl,
    language=None,
    audio_root=None,
    check_audio=False,
    append=False,
):
    """将单个 TRS 文件转换成 CustomDataset 使用的 JSONL。"""
    convert_trs_files(
        [Path(trs_file)], output_jsonl, language, audio_root, check_audio, append
    )


def convert_trs_directory(
    trs_dir,
    output_jsonl,
    language=None,
    audio_root=None,
    check_audio=False,
    append=False,
):
    """将目录下面所有 .trs 合并成一个 JSONL。"""
    trs_files = sorted(Path(trs_dir).rglob("*.trs"))
    convert_trs_files(
        trs_files, output_jsonl, language, audio_root, check_audio, append
    )


def main():
    parser = argparse.ArgumentParser(
        description="将 TRS 数据转换为 Whisper CustomDataset 使用的 JSONL"
    )
    parser.add_argument(
        "--input", required=True, help="输入的 .trs 文件或者包含 .trs 的目录"
    )
    parser.add_argument(
        "--output", required=True, help="输出 JSONL 文件，例如 test.json"
    )
    parser.add_argument("--language", default=None, help="语言，例如 en / zh / ja")
    parser.add_argument("--audio_root", default=None, help="音频路径根目录，可选")
    parser.add_argument(
        "--check_audio", action="store_true", help="检查音频文件是否实际存在"
    )
    parser.add_argument(
        "--append", action="store_true", help="追加到输出文件；默认覆盖"
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        convert = convert_trs_file
    elif os.path.isdir(args.input):
        convert = convert_trs_directory
    else:
        raise FileNotFoundError(f"输入不存在: {args.input}")

    convert(
        args.input,
        args.output,
        language=args.language,
        audio_root=args.audio_root,
        check_audio=args.check_audio,
        append=args.append,
    )


if __name__ == "__main__":
    main()