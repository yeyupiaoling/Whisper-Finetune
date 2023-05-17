import argparse
import functools
import os
import time

from faster_whisper import WhisperModel
from flask import request, Flask, render_template
from flask_cors import CORS

from utils.utils import add_arguments, print_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",             str,    "0.0.0.0",            "监听主机的IP地址")
add_arg("port",             int,    5000,                 "服务所使用的端口号")
add_arg("audio_path",  type=str,  default="dataset/test.wav",        help="预测的音频路径")
add_arg("model_path",  type=str,  default="models/whisper-tiny-ct2", help="转换后的模型路径，转换方式看文档")
add_arg("language",    type=str, default="zh",    help="设置语言")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("use_int8",    type=bool, default=False,  help="是否使用int8进行预测")
add_arg("beam_size",   type=int,  default=10,     help="解码搜索大小")
add_arg("num_workers", type=int,  default=1,      help="预测器的并发数量")
add_arg("vad_filter",  type=bool, default=False,  help="是否使用VAD过滤掉部分没有讲话的音频")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# 允许跨越访问
CORS(app)

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


# 语音识别接口
@app.route("/transcribe", methods=['POST'])
def transcribe():
    f = request.files['audio']
    if f:
        try:
            start = time.time()
            # 执行识别
            segments, info = model.transcribe(f, beam_size=args.beam_size, language=args.language,
                                              vad_filter=args.vad_filter)
            result_text = ''
            for segment in segments:
                text = segment.text
                result_text += text
            end = time.time()
            print(f"识别时间：{round((end - start) * 1000)}ms，识别结果：{result_text}")
            result = str({"code": 0, "msg": "success", "result": result_text}).replace("'", '"')
            return result
        except:
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
