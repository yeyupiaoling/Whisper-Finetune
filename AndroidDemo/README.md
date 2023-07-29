# Android部署项目

简体中文 | [English](./README_en.md)

## 转换模型

1. To start by cloning the Whisper native source code, as we need some of its files, run the following command from the root of the `whisper-finetune` project.
```shell
git clone https://github.com/openai/whisper.git
```

2. 然后开始转换模型，请在`Whisper-Finetune`项目根目录下执行`convert-ggml.py`程序，把模型转换为Android项目所需的ggml格式的模型，需要转换的模型可以是原始的Transformers模型，也可以是微调的模型。
```shell
python convert-ggml.py --model_dir=models/whisper-tiny-finetune/ --whisper_dir=whisper/ --output_path=models/ggml-model.bin
```

3. 把模型放在Android项目的`app/src/main/assets/models`目录下，然后就可以使用Android Studio打开项目了。


## 编译说明

1. 默认使用的NDK版本是`25.2.9519653`，如果下面修改其他版本，要修改`app/build.gradle`里面的配置。
2. **注意，在真正使用时，一定要发布`release`的APK包，这样推理速度才快。**
3. 本项目已经发布了`release`的APK包，请在`Whisper-Finetune`项目主页的最后扫码下载。

## 效果图

效果图如下，这里使用的模型是量化为半精度tiny模型，准确率不高。
<br/>
<div align="center">
<img src="../docs/images/android2.jpg" alt="Android效果图" width="200">
<img src="../docs/images/android1.jpg" alt="Android效果图" width="200">
<img src="../docs/images/android3.jpg" alt="Android效果图" width="200">
<img src="../docs/images/android4.jpg" alt="Android效果图" width="200">
</div>

## 下载安装包

可以点击这里下载[Android安装包](https://yeyupiaoling.cn/whisper.apk)，注意，为了安装包小，这里使用的模型是量化为半精度tiny模型，准确率不高，如果想更换模型的，请执行编译项目。
<br/>
<div align="center">
<img src="../docs/images/android.jpg" alt="Android安装包" width="200">
</div>
