import _thread
import argparse
import functools
import os
import platform
import time
import tkinter.messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename

import numpy as np
import soundcard
import soundfile
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
from zhconv import convert

from utils.utils import print_arguments, add_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune/", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("language",    type=str,  default="chinese", help="设置语言，如果为None则预测的是多语言")
add_arg("use_gpu",     type=bool, default=True,      help="是否使用gpu进行预测")
add_arg("num_beams",   type=int,  default=1,         help="解码搜索大小")
add_arg("batch_size",  type=int,  default=16,        help="预测batch_size大小")
add_arg("use_compile", type=bool, default=False,     help="是否使用Pytorch2.0的编译器")
add_arg("task",        type=str,  default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("assistant_model_path",  type=str, default=None, help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
add_arg("local_files_only",      type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
add_arg("use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
args = parser.parse_args()
print_arguments(args)


class SpeechRecognitionApp:
    def __init__(self, window: Tk, args):
        self.window = window
        self.wav_path = None
        self.predicting = False
        self.playing = False
        self.recording = False
        # 录音参数
        self.frames = []
        self.sample_rate = 16000
        self.interval_time = 0.5
        self.block_size = int(self.sample_rate * self.interval_time)
        # 最大录音时长
        self.max_record = 600
        # 录音保存的路径
        self.output_path = 'dataset/record'
        # 指定窗口标题
        self.window.title("夜雨飘零语音识别")
        # 固定窗口大小
        self.window.geometry('870x500')
        self.window.resizable(False, False)
        # 识别短语音按钮
        self.short_button = Button(self.window, text="选择文件", width=20, command=self.predict_audio_thread)
        self.short_button.place(x=10, y=10)
        # 录音按钮
        self.record_button = Button(self.window, text="录音识别", width=20, command=self.record_audio_thread)
        self.record_button.place(x=170, y=10)
        # 播放音频按钮
        self.play_button = Button(self.window, text="播放音频", width=20, command=self.play_audio_thread)
        self.play_button.place(x=330, y=10)
        # 输出结果文本框
        self.result_label = Label(self.window, text="输出日志：")
        self.result_label.place(x=10, y=70)
        self.result_text = Text(self.window, width=120, height=30)
        self.result_text.place(x=10, y=100)
        # 转阿拉伯数字控件
        self.check_frame = Frame(self.window)
        self.joint_text_check_var = BooleanVar()
        self.joint_text_check = Checkbutton(self.check_frame, text='拼接文本', variable=self.joint_text_check_var)
        self.joint_text_check.grid(column=0, row=0)
        self.to_simple_check_var = BooleanVar()
        self.to_simple_check = Checkbutton(self.check_frame, text='繁体转简体', variable=self.to_simple_check_var)
        self.to_simple_check.grid(column=1, row=0)
        self.to_simple_check.select()
        self.task_check_var = BooleanVar()
        self.task_check = Checkbutton(self.check_frame, text='音频转录', variable=self.task_check_var)
        self.task_check.grid(column=2, row=0)
        self.task_check.select()
        self.check_frame.grid(row=1)
        self.check_frame.place(x=600, y=10)

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
        if args.use_bettertransformer and not args.use_flash_attention_2:
            model = model.to_bettertransformer()
        # 使用Pytorch2.0的编译器
        if args.use_compile:
            if torch.__version__ >= "2" and platform.system().lower() != 'windows':
                model = torch.compile(model)
        model.to(device)

        # 获取助手模型
        generate_kwargs_pipeline = None
        if args.assistant_model_path is not None:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                args.assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            assistant_model.to(device)
            generate_kwargs_pipeline = {"assistant_model": assistant_model}

        # 获取管道
        self.infer_pipe = pipeline("automatic-speech-recognition",
                                   model=model,
                                   tokenizer=processor.tokenizer,
                                   feature_extractor=processor.feature_extractor,
                                   max_new_tokens=128,
                                   chunk_length_s=30,
                                   batch_size=2,
                                   torch_dtype=torch_dtype,
                                   generate_kwargs=generate_kwargs_pipeline,
                                   device=device)

        # 预热
        _ = self.infer_pipe("dataset/test.wav")

    # 预测短语音线程
    def predict_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")],
                                            initialdir='./dataset')
            if self.wav_path == '': return
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, "已选择音频文件：%s\n" % self.wav_path)
            self.result_text.insert(END, "正在识别中...\n")
            _thread.start_new_thread(self.predict_audio, (self.wav_path,))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测短语音
    def predict_audio(self, wav_path):
        self.predicting = True
        self.result_text.delete('1.0', 'end')
        try:
            task = "transcribe" if self.task_check_var.get() else "translate"
            # 推理参数
            generate_kwargs = {"task": task, "num_beams": args.num_beams}
            if args.language is not None:
                generate_kwargs["language"] = args.language
            # 推理
            result = self.infer_pipe(wav_path, return_timestamps=True, generate_kwargs=generate_kwargs)
            # 判断是否要分段输出
            if self.joint_text_check_var.get():
                text = result['text']
                # 繁体转简体
                if self.to_simple_check_var.get():
                    text = convert(text, 'zh-cn')
                self.result_text.delete('1.0', 'end')
                self.result_text.insert(END, f"{text}\n")
            else:
                for chunk in result["chunks"]:
                    text = chunk['text']
                    # 繁体转简体
                    if self.to_simple_check_var.get():
                        text = convert(text, 'zh-cn')
                    self.result_text.insert(END, f"[{chunk['timestamp'][0]} - {chunk['timestamp'][1]}]：{text}\n")
            self.predicting = False
        except Exception as e:
            print(e)
            self.predicting = False

    # 录音识别线程
    def record_audio_thread(self):
        if not self.playing and not self.recording:
            self.result_text.delete('1.0', 'end')
            self.recording = True
            _thread.start_new_thread(self.record_audio, ())
        else:
            if self.playing:
                tkinter.messagebox.showwarning('警告', '正在播放音频，无法录音！')
            else:
                # 停止播放
                self.recording = False

    # 播放音频线程
    def play_audio_thread(self):
        if self.wav_path is None or self.wav_path == '':
            tkinter.messagebox.showwarning('警告', '音频路径为空！')
        else:
            if not self.playing and not self.recording:
                _thread.start_new_thread(self.play_audio, ())
            else:
                if self.recording:
                    tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
                else:
                    # 停止播放
                    self.playing = False

    def record_audio(self):
        self.frames = []
        self.record_button.configure(text='停止录音')
        self.result_text.insert(END, "正在录音...\n")
        # 打开默认的输入设备
        input_device = soundcard.default_microphone()
        recorder = input_device.recorder(samplerate=self.sample_rate, channels=1, blocksize=self.block_size)
        with recorder:
            while True:
                if len(self.frames) * self.interval_time > self.max_record: break
                # 开始录制并获取数据
                data = recorder.record(numframes=self.block_size)
                data = data.squeeze()
                self.frames.append(data)
                self.result_text.delete('1.0', 'end')
                self.result_text.insert(END, f"已经录音{len(self.frames) * self.interval_time}秒\n")
                if not self.recording: break
        # 拼接录音数据
        data = np.concatenate(self.frames)
        # 保存音频数据
        os.makedirs(self.output_path, exist_ok=True)
        self.wav_path = os.path.join(self.output_path, '%s.wav' % str(int(time.time())))
        soundfile.write(self.wav_path, data=data, samplerate=self.sample_rate)
        self.recording = False
        self.record_button.configure(text='录音识别')
        self.result_text.delete('1.0', 'end')
        _thread.start_new_thread(self.predict_audio, (self.wav_path,))

    # 播放音频
    def play_audio(self):
        self.play_button.configure(text='停止播放')
        self.playing = True
        default_speaker = soundcard.default_speaker()
        data, sr = soundfile.read(self.wav_path)
        with default_speaker.player(samplerate=sr) as player:
            for i in range(0, data.shape[0], sr):
                if not self.playing: break
                d = data[i:i + sr]
                player.play(d / np.max(np.abs(d)))
        self.playing = False
        self.play_button.configure(text='播放音频')


tk = Tk()
myapp = SpeechRecognitionApp(tk, args)

if __name__ == '__main__':
    tk.mainloop()
