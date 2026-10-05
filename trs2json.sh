#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# 每个 TRS 输出一个同名 JSON（JSONL 格式），重复运行覆盖对应文件。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trs_dir=/ctd/SpeechData/Trainset/English/Others/LibriSpeech/16k/trs/20241126/original
audio_base_path=/ctd/SpeechData/Trainset/English
output_dir=dataset/trainset/LibriSpeech
mkdir -p -- "${output_dir}"

# 只读取指定目录顶层的 TRS，不递归读取子目录。
trs_files=("${trs_dir}"/*.trs)
if (( ${#trs_files[@]} == 0 )); then
    echo "未找到 TRS 文件：${trs_dir}/*.trs" >&2
    exit 1
fi

for trs_file in "${trs_files[@]}"; do
    [[ -f "${trs_file}" ]] || continue
    filename=${trs_file##*/}
    output="${output_dir}/${filename%.trs}.json"

    python trs2json.py \
        --input "${trs_file}" \
        --output "${output}" \
        --language en \
        --audio_root "${audio_base_path}"
done
