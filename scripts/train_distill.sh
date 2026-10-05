#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 单机 DDP：默认使用所有可见 GPU，每卡一个进程、一份冻结教师和学生。
# 指定两张卡：CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_distill.sh
# 限定进程数：CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=2 bash scripts/train_distill.sh
# 断点续训：bash scripts/train_distill.sh --resume_from_checkpoint output/distillation/checkpoint-1000
# 只打印启动命令（不加载模型）：DRY_RUN=1 bash scripts/train_distill.sh
# 使用当前激活环境中的 torchrun；先激活安装了训练依赖的环境。
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
if [[ "$NPROC_PER_NODE" != gpu && ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    printf '错误：NPROC_PER_NODE 必须是正整数或 gpu。\n' >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 目录会按文件名排序合并当前层全部 .json 清单，不递归扫描。
# 训练：train-clean-100 + train-clean-360 + train-other-500。
# 验证：dev-clean + dev-other；testset 留给独立评测，避免用于选择最佳模型。
# JSON 中的音频绝对路径必须在训练机器上可访问。
TRAIN_DATA="${TRAIN_DATA:-dataset/trainset/LibriSpeech}"
TEST_DATA="${TEST_DATA:-dataset/devset/LibriSpeech}"

# large-v2 教师 -> small 学生；STUDENT_MODEL='' 使用随机初始化的本地 Transformer。
STUDENT_MODEL="${STUDENT_MODEL-openai/whisper-small}"
student_args=()
if [[ -n "$STUDENT_MODEL" ]]; then
    student_args=(--student_model "$STUDENT_MODEL")
fi

# batch size / workers 均为每个 GPU 进程的配置。
# 默认有效全局 batch = 4 × 4 × GPU 数量；显存不足可降低单卡 batch 并提高累积步数。
# --standalone 自动选择 rendezvous 端口，避免并发单机任务的端口冲突。
command=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    train_distil.py \
    --teacher_model "${TEACHER_MODEL:-openai/whisper-large-v2}" \
    "${student_args[@]}" \
    --train_data "$TRAIN_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "${OUTPUT_DIR:-output/distillation}" \
    --language en \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --fp16 True \
    --bf16 False \
    --timestamps False \
    --min_audio_len 0.5 \
    --max_audio_len 30 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 100 \
    --lambda 0.5 \
    --temperature 1.5 \
    "$@")

# 末尾参数优先于上述默认值；BF16 使用 --fp16 False --bf16 True。
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<all>}"
printf '启动命令：'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi
exec "${command[@]}"