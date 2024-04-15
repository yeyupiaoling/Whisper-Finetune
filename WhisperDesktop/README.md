# Windows桌面应用

简体中文 | [English](./README_en.md)

该程序是使用[Whisper](https://github.com/Const-me/Whisper)翻译得的，源码可以前面该项目查看。该程序使用的模型格式是GGML格式，跟Android部署的一样，所以需要转换模型格式才能使用。

## 转换模型

1. 然后开始转换模型，请在`Whisper-Finetune`项目根目录下执行`convert-ggml.py`程序，把模型转换为Android项目所需的ggml格式的模型，需要转换的模型可以是原始的Transformers模型，也可以是微调的模型。
```shell
python convert-ggml.py --model_dir=models/whisper-tiny-finetune/ --output_path=models/whisper-tiny-finetune-ggml.bin
```


## 效果图

效果图如下：
<br/>
<div align="center">
<img src="../docs/images/desktop1.jpg" alt="Windows桌面应用效果图"><br/>
图1：加载模型页面
<br/>
<img src="../docs/images/desktop2.jpg" alt="Windows桌面应用效果图"><br/>
图2：选择音频文件转录
<br/>
<img src="../docs/images/desktop3.jpg" alt="Windows桌面应用效果图"><br/>
图3：录音转录
</div>
