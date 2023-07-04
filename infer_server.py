import argparse
import asyncio
import functools
import json
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, Body, UploadFile, Request
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from zhconv import convert

from utils.data_utils import remove_punctuation
from utils.utils import add_arguments, print_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",        type=str,  default="0.0.0.0", help="监听主机的IP地址")
add_arg("port",        type=int,  default=5000,        help="服务所使用的端口号")
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune-ct2", help="转换后的模型路径，转换方式看文档")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("use_int8",    type=bool, default=False,  help="是否使用int8进行预测")
add_arg("beam_size",   type=int,  default=10,     help="解码搜索大小")
add_arg("num_workers", type=int,  default=2,      help="预测器的并发数量")
add_arg("vad_filter",  type=bool, default=True,  help="是否使用VAD过滤掉部分没有讲话的音频")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.model_path), f"模型文件{args.model_path}不存在"
# 加载模型
if args.use_gpu:
    if not args.use_int8:
        model = WhisperModel(args.model_path, device="cuda", compute_type="float16", num_workers=args.num_workers,
                             local_files_only=args.local_files_only)
    else:
        model = WhisperModel(args.model_path, device="cuda", compute_type="int8_float16", num_workers=args.num_workers,
                             local_files_only=args.local_files_only)
else:
    model = WhisperModel(args.model_path, device="cpu", compute_type="int8", num_workers=args.num_workers,
                         local_files_only=args.local_files_only)
# 预热
_, _ = model.transcribe("dataset/test.wav", beam_size=5)

app = FastAPI(title="夜雨飘零语音识别")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")
model_semaphore = None


def release_model_semaphore():
    model_semaphore.release()


def recognition(file: File, to_simple: int, remove_pun: int, language: str = "zh", task: str = "transcribe"):
    segments, info = model.transcribe(file, beam_size=10, task=task, language=language, vad_filter=args.vad_filter)
    for segment in segments:
        text = segment.text
        if to_simple == 1:
            text = convert(text, 'zh-cn')
        if remove_pun == 1:
            text = remove_punctuation(text)
        ret = {"result": text, "start": round(segment.start, 2), "end": round(segment.end, 2)}
        # 返回json格式的字节
        yield json.dumps(ret).encode() + b"\0"


@app.post("/recognition_stream")
async def api_recognition_stream(to_simple: int = Body(1, description="是否繁体转简体", embed=True),
                                 remove_pun: int = Body(0, description="是否删除标点符号", embed=True),
                                 language: str = Body("zh", description="设置语言，简写，如果不指定则自动检测语言", embed=True),
                                 task: str = Body("transcribe", description="识别任务类型，支持transcribe和translate", embed=True),
                                 audio: UploadFile = File(..., description="音频文件")):
    global model_semaphore
    if language == "None": language = None
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(5)
    await model_semaphore.acquire()
    contents = await audio.read()
    data = BytesIO(contents)
    generator = recognition(file=data, to_simple=to_simple, remove_pun=remove_pun, language=language, task=task)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/recognition")
async def api_recognition(to_simple: int = Body(1, description="是否繁体转简体", embed=True),
                          remove_pun: int = Body(0, description="是否删除标点符号", embed=True),
                          language: str = Body("zh", description="设置语言，简写，如果不指定则自动检测语言", embed=True),
                          task: str = Body("transcribe", description="识别任务类型，支持transcribe和translate", embed=True),
                          audio: UploadFile = File(..., description="音频文件")):
    if language == "None":language=None
    contents = await audio.read()
    data = BytesIO(contents)
    generator = recognition(file=data, to_simple=to_simple, remove_pun=remove_pun, language=language, task=task)
    results = []
    for output in generator:
        output = json.loads(output[:-1].decode("utf-8"))
        results.append(output)
    ret = {"results": results, "code": 0}
    return ret


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)
