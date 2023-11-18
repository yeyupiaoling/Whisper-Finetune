#!/bin/bash

# Transformer模型
python compute_speed_tf.py --model_path=openai/whisper-tiny
python compute_speed_tf.py --model_path=openai/whisper-tiny --use_compile=True
python compute_speed_tf.py --model_path=openai/whisper-tiny --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-tiny --use_flash_attention_2=True
python compute_speed_tf.py --model_path=openai/whisper-tiny --use_compile=True --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-tiny --use_compile=True --use_flash_attention_2=True

python compute_speed_tf.py --model_path=openai/whisper-base
python compute_speed_tf.py --model_path=openai/whisper-base --use_compile=True
python compute_speed_tf.py --model_path=openai/whisper-base --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-base --use_flash_attention_2=True
python compute_speed_tf.py --model_path=openai/whisper-base --use_compile=True --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-base --use_compile=True --use_flash_attention_2=True

python compute_speed_tf.py --model_path=openai/whisper-small
python compute_speed_tf.py --model_path=openai/whisper-small --use_compile=True
python compute_speed_tf.py --model_path=openai/whisper-small --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-small --use_flash_attention_2=True
python compute_speed_tf.py --model_path=openai/whisper-small --use_compile=True --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-small --use_compile=True --use_flash_attention_2=True

python compute_speed_tf.py --model_path=openai/whisper-medium
python compute_speed_tf.py --model_path=openai/whisper-medium --use_compile=True
python compute_speed_tf.py --model_path=openai/whisper-medium --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-medium --use_flash_attention_2=True
python compute_speed_tf.py --model_path=openai/whisper-medium --use_compile=True --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-medium --use_compile=True --use_flash_attention_2=True

python compute_speed_tf.py --model_path=openai/whisper-large-v2
python compute_speed_tf.py --model_path=openai/whisper-large-v2 --use_compile=True
python compute_speed_tf.py --model_path=openai/whisper-large-v2 --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-large-v2 --use_flash_attention_2=True
python compute_speed_tf.py --model_path=openai/whisper-large-v2 --use_compile=True --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-large-v2 --use_compile=True --use_flash_attention_2=True

python compute_speed_tf.py --model_path=openai/whisper-large-v3
python compute_speed_tf.py --model_path=openai/whisper-large-v3 --use_compile=True
python compute_speed_tf.py --model_path=openai/whisper-large-v3 --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-large-v3 --use_flash_attention_2=True
python compute_speed_tf.py --model_path=openai/whisper-large-v3 --use_compile=True --use_bettertransformer=True
python compute_speed_tf.py --model_path=openai/whisper-large-v3 --use_compile=True --use_flash_attention_2=True

# Ctranslate2模型
python compute_speed_ct2.py --model_path=../models/whisper-tiny-ct2/
python compute_speed_ct2.py --model_path=../models/whisper-base-ct2/
python compute_speed_ct2.py --model_path=../models/whisper-small-ct2/
python compute_speed_ct2.py --model_path=../models/whisper-medium-ct2/
python compute_speed_ct2.py --model_path=../models/whisper-large-v2-ct2/
python compute_speed_ct2.py --model_path=../models/whisper-large-v3-ct2/

python compute_speed_ct2.py --model_path=../models/whisper-tiny-ct2/  --use_int8=True
python compute_speed_ct2.py --model_path=../models/whisper-base-ct2/  --use_int8=True
python compute_speed_ct2.py --model_path=../models/whisper-small-ct2/  --use_int8=True
python compute_speed_ct2.py --model_path=../models/whisper-medium-ct2/  --use_int8=True
python compute_speed_ct2.py --model_path=../models/whisper-large-v2-ct2/  --use_int8=True
python compute_speed_ct2.py --model_path=../models/whisper-large-v3-ct2/  --use_int8=True
