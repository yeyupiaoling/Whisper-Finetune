# Android部署项目

## 转换模型

1. 首先要克隆Whisper原生的源码，因为需要它的一些文件，请在`Whisper-Finetune`项目根目录下执行下面命令。
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