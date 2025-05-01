import argparse
import functools
import os
import platform

import torch
import uvicorn
from fastapi import FastAPI, File, Body, UploadFile, Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
from zhconv import convert

from utils.data_utils import remove_punctuation
from utils.utils import add_arguments, print_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",        type=str,  default="0.0.0.0",   help="监听主机的IP地址")
add_arg("port",        type=int,  default=5000,        help="服务所使用的端口号")
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune/", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("num_beams",   type=int,  default=1,      help="解码搜索大小")
add_arg("batch_size",  type=int,  default=16,     help="预测batch_size大小")
add_arg("use_compile", type=bool, default=False,  help="是否使用Pytorch2.0的编译器")
add_arg("assistant_model_path",  type=str,  default=None,  help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
add_arg("local_files_only",      type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
add_arg("use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
args = parser.parse_args()
print_arguments(args)

# 设置设备
device = "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() and args.use_gpu else torch.float32

# 获取Whisper的特征提取器、编码器和解码器
processor = AutoProcessor.from_pretrained(args.model_path)

# 获取模型
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    use_flash_attention_2=args.use_flash_attention_2
)
model.generation_config.forced_decoder_ids = None
if args.use_bettertransformer and not args.use_flash_attention_2:
    model = model.to_bettertransformer()
# 使用Pytorch2.0的编译器
if args.use_compile:
    if torch.__version__ >= "2" and platform.system().lower() != 'windows':
        model = torch.compile(model)
model.to(device)

# 获取助手模型
generate_kwargs_pipeline = {"max_new_tokens": 128}
if args.assistant_model_path is not None:
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    assistant_model.to(device)
    generate_kwargs_pipeline = {"assistant_model": assistant_model}

# 获取管道
infer_pipe = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      chunk_length_s=30,
                      batch_size=args.batch_size,
                      torch_dtype=torch_dtype,
                      generate_kwargs=generate_kwargs_pipeline,
                      device=device)

# 预热
_ = infer_pipe("dataset/test.wav")

app = FastAPI(title="夜雨飘零语音识别")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")


def recognition(file: File, to_simple: int, remove_pun: int, language: str = None, task: str = "transcribe"):
    # 推理参数
    generate_kwargs = {"task": task, "num_beams": args.num_beams}
    if language is not None:
        generate_kwargs["language"] = args.language
    # 推理
    result = infer_pipe(file, return_timestamps=True, generate_kwargs=generate_kwargs)
    results = []
    for chunk in result["chunks"]:
        text = chunk['text']
        if to_simple == 1:
            text = convert(text, 'zh-cn')
        if remove_pun == 1:
            text = remove_punctuation(text)
        ret = {"text": text, "start": chunk['timestamp'][0], "end": chunk['timestamp'][1]}
        results.append(ret)
    return results


@app.post("/recognition")
async def api_recognition(to_simple: int = Body(1, description="是否繁体转简体", embed=True),
                          remove_pun: int = Body(0, description="是否删除标点符号", embed=True),
                          language: str = Body(None, description="设置语言，如果为None则预测的是多语言", embed=True),
                          task: str = Body("transcribe", description="识别任务类型，支持transcribe和translate", embed=True),
                          audio: UploadFile = File(..., description="音频文件")):
    if language == "None": language = None
    data = await audio.read()
    results = recognition(file=data, to_simple=to_simple, remove_pun=remove_pun, language=language, task=task)
    ret = {"results": results, "code": 0}
    return ret


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)
