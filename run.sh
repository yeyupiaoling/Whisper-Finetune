#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --use_8bit=False --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-base --use_8bit=False --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-base/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-small --use_8bit=True --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-small/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-medium --use_8bit=True --per_device_train_batch_size=4 --per_device_eval_batch_size=2 --gradient_accumulation_steps=2
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-medium/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-large-v2 --use_8bit=True --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --gradient_accumulation_steps=4
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-large-v2/checkpoint-final

CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-tiny-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-base-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-small-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-medium-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-large-v2-finetune
