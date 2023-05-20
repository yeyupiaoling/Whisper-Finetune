import _thread
import argparse
import functools
import os
import time
import tkinter.messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename

import numpy as np
import soundcard
import soundfile
from faster_whisper import WhisperModel
from zhconv import convert

from utils.utils import print_arguments, add_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune-ct2", help="转换后的模型路径，转换方式看文档")
add_arg("language",    type=str, default="zh",    help="设置语言，必须简写，如果为None则自动检测语言")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("use_int8",    type=bool, default=False,  help="是否使用int8进行预测")
add_arg("beam_size",   type=int,  default=10,     help="解码搜索大小")
add_arg("vad_filter",  type=bool, default=True,  help="是否使用VAD过滤掉部分没有讲话的音频")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
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

        # 检查模型文件是否存在
        assert os.path.exists(args.model_path), f"模型文件{args.model_path}不存在"
        # 加载模型
        if args.use_gpu:
            if not args.use_int8:
                self.model = WhisperModel(args.model_path, device="cuda", compute_type="float16",
                                          local_files_only=args.local_files_only)
            else:
                self.model = WhisperModel(args.model_path, device="cuda", compute_type="int8_float16",
                                          local_files_only=args.local_files_only)
        else:
            self.model = WhisperModel(args.model_path, device="cpu", compute_type="int8",
                                      local_files_only=args.local_files_only)
        # 预热
        _, _ = self.model.transcribe("dataset/test.wav", beam_size=5)

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
            segments, info = self.model.transcribe(wav_path, beam_size=args.beam_size, language=args.language,
                                                   vad_filter=args.vad_filter, task=task)
            result_text = ''
            for segment in segments:
                text = segment.text
                # 繁体转简体
                if self.to_simple_check_var.get():
                    text = convert(text, 'zh-cn')
                # 判断是否要分段输出
                if self.joint_text_check_var.get():
                    result_text += text
                    self.result_text.delete('1.0', 'end')
                    self.result_text.insert(END, f"{result_text}\n")
                else:
                    self.result_text.insert(END, f"[{round(segment.start, 2)} - {round(segment.end, 2)}]：{text}\n")
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
