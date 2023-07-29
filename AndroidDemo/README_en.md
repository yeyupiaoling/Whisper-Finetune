# Android

[简体中文](./README.md) | English

**Disclaimer, this document was obtained through machine translation, please check the original document [here](./README.md).**


## Convert model

1. To start by cloning the Whisper native source code, as we need some of its files, run the following command from the root of the `whisper-finetune` project.
```shell
git clone https://github.com/openai/whisper.git
```

2. To convert your models, run `convert-ggml.py` from the root of your `Whisper-Finetune` project to convert your models to ggml format for your Android project. The models you need to convert can be original Transformers. It can also be a fine-tuned model.
```shell
python convert-ggml.py --model_dir=models/whisper-tiny-finetune/ --whisper_dir=whisper/ --output_path=models/ggml-model.bin
```

3. Put the model in the Android project `app/SRC/main/assets/models` directory, and then you can use the Android open Studio project.


## Build notes

1. The default NDK version used is `25.2.9519653`, if you change the other version below, you will need to change the configuration in `app/build.gradle`.
2. **Note that in real use, be sure to release the `release` APK package so that inference is fast.**
3. This project has released the `release` APK package, please scan the code at the end of the `Whisker-finetune` project homepage to download it.

## Effect picture

The effect picture is as follows. The model used here is quantized as a half-precision tiny model, which has a low accuracy.
<br/>
<div align="center">
<img src="../docs/images/android2.jpg" alt="Android效果图" width="200">
<img src="../docs/images/android1.jpg" alt="Android效果图" width="200">
<img src="../docs/images/android3.jpg" alt="Android效果图" width="200">
<img src="../docs/images/android4.jpg" alt="Android效果图" width="200">
</div>

## Download APK

Can click here to download the [Android APK](https://yeyupiaoling.cn/whisper.apk), note that in order to install package is small, quantitative model used here for half a tiny model precision and accuracy is not high, if you want to change model, please compile the project execution.

<br/>
<div align="center">
<img src="../docs/images/android.jpg" alt="Android安装包" width="200">
</div>
