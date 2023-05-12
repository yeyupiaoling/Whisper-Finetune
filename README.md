# 微调Whisper模型和加速推理

## 前言

OpenAI在开源了号称其英文语音辨识能力已达到人类水准的Whisper项目，且它亦支持其它98种语言的自动语音辨识。Whisper所提供的自动语音识与翻译任务，它们能将各种语言的语音变成文本，也能将这些文本翻译成英文。本项目主要的目的是为了对Whisper模型使用Lora进行微调，目前开源了好几个模型，具体可以在[openai](https://huggingface.co/openai)查看，下面列出了常用的几个模型。另外项目最后还对语音识别加速推理，使用了CTranslate2加速推理，提示一下，加速推理支持直接使用Whisper原模型转换，并不一定需要微调。

 - openai/whisper-tiny
 - openai/whisper-base
 - openai/whisper-small
 - openai/whisper-medium
 - openai/whisper-large
 - openai/whisper-large-v2

**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`758170167`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>


使用环境：

- Anaconda 3
- Python 3.8
- Pytorch 1.12.1
- Ubuntu 18.04
- GPU A100-PCIE-40GB*1


## 项目主要程序介绍

1. `aishell.py`：制作AIShell训练数据。
2. `finetune.py`：微调模型。
3. `merge_lora.py`：合并Whisper和Lora的模型。
4. `evaluation.py`：评估使用微调后的模型或者Whisper原模型。
5. `infer.py`：使用微调后的模型或者Whisper原模型预测。
6. `infer_ct2.py`：使用转换的模型预测。

## 模型测试表

1. 原始模型字错率测试表

|       使用模型       | 语言  | aishell_test(CER) | test_net(CER) | test_meeting(CER) | 下载地址 |
|:----------------:|:---:|:-----------------:|:-------------:|:-----------------:|:----:|
|   whisper-tiny   | 普通话 |                   |               |                   |      |
|   whisper-base   | 普通话 |                   |               |                   |      |
|  whisper-small   | 普通话 |                   |               |                   |      |
|  whisper-medium  | 普通话 |                   |               |                   |      |
| whisper-large-v2 | 普通话 |                   |               |                   |      |


2. 微调[WenetSpeech](https://github.com/yeyupiaoling/PPASR/blob/develop/docs/wenetspeech.md)数据集后字错率测试表

|       使用模型       | 语言  |                                         微调数据集                                         | aishell_test(CER) | test_net(CER) | test_meeting(CER) | 下载地址 |
|:----------------:|:---:|:-------------------------------------------------------------------------------------:|:-----------------:|:-------------:|:-----------------:|:----:|
|   whisper-tiny   | 普通话 | [WenetSpeech](https://github.com/yeyupiaoling/PPASR/blob/develop/docs/wenetspeech.md) |                   |               |                   |      |
|   whisper-base   | 普通话 | [WenetSpeech](https://github.com/yeyupiaoling/PPASR/blob/develop/docs/wenetspeech.md) |                   |               |                   |      |
|  whisper-small   | 普通话 | [WenetSpeech](https://github.com/yeyupiaoling/PPASR/blob/develop/docs/wenetspeech.md) |                   |               |                   |      |
|  whisper-medium  | 普通话 | [WenetSpeech](https://github.com/yeyupiaoling/PPASR/blob/develop/docs/wenetspeech.md) |                   |               |                   |      |
| whisper-large-v2 | 普通话 | [WenetSpeech](https://github.com/yeyupiaoling/PPASR/blob/develop/docs/wenetspeech.md) |                   |               |                   |      |

3. 未加速和加速后的推理速度测试表

|       使用模型       | 原生模型实时率 | 转换CTranslate2加速后实时率 |
|:----------------:|:-------:|:-------------------:|
|   whisper-tiny   |         |                     |
|   whisper-base   |         |                     |    
|  whisper-small   |         |                     | 
|  whisper-medium  |         |                     |  
| whisper-large-v2 |         |                     |

**重要说明：**
1. 在评估的时候移除模型输出的标点符号，并把繁体中文转成简体中文。
2. RTF= 所有音频总时间(单位秒) / ASR识别所有音频处理时间(单位秒)

## 安装环境

- 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

- 安装所需的依赖库。

```shell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 准备数据

训练的数据集如下，是一个JSON的数据列表。Whisper是支持有标点符号的，所以训练的数据集中可以带有标点符号。本项目提供了一个制作AIShell数据集的程序`aishell.py`，执行这个程序可以自动下载并生成如下列格式的训练集和测试集，**注意：** 这个程序可以通过指定AIShell的压缩文件来跳过下载过程的，如果直接下载会非常慢，可以使用一些如迅雷等下载器下载该数据集，然后通过参数`--filepath`指定下载的压缩文件路径，如`/home/test/data_aishell.tgz`。
```json
[
    {
        "audio": {
            "path": "dataset/audio/data_aishell/wav/test/S0764/BAC009S0764W0489.wav"
        },
        "duration": 3.97,
        "sentence": "不是她的戏或是她的八卦"
    },
    {
        "audio": {
            "path": "dataset/audio/data_aishell/wav/test/S0764/BAC009S0764W0202.wav"
        },
        "duration": 5.63,
        "sentence": "第二批三网融合试点工作业已启动"
    }
]
```

## 微调模型

准备好数据之后，就可以开始微调模型了。训练最重要的两个参数分别是，`--base_model`指定微调的Whisper模型，这个参数值需要在[HuggingFace](https://huggingface.co/openai)存在的，这个不需要提前下载，启动训练时可以自动下载。第二个`--output_path`是是训练时保存的Lora检查点路径，因为我们使用Lora来微调模型。其他更多的参数请查看这个程序。

### 单卡训练

单卡训练命令如下，Windows系统可以不添加`CUDA_VISIBLE_DEVICES`参数。
```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-tiny --output_dir=models/whisper-tiny-lora
```

### 多卡训练

多卡训练有两种方法，分别是torchrun和accelerate，开发者可以根据自己的习惯使用对应的方式。

1. 使用torchrun启动多卡训练，命令如下，通过`--nproc_per_node`指定使用的显卡数量。
```shell
torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --output_dir=models/whisper-tiny-lora
```

2. 使用accelerate启动多卡训练，如果是第一次使用accelerate，要配置训练参数，方式如下。

首先配置训练参数，过程是让开发者回答几个问题，基本都是默认就可以，但有几个参数需要看实际情况设置。
```shell
accelerate config
```

大概过程就是这样：
```
----------------------------------In which compute environment are you running?
This machine
----------------------------------Which type of machine are you using? 
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]: 
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
----------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp16 
```

配置完成之后，可以使用以下命令查看配置。
```shell
accelerate env
```

开始训练命令如下。
```shell
accelerate launch finetune.py --base_model=openai/whisper-tiny --output_dir=models/whisper-tiny-lora
```


输出日志如下：
```shell
{'loss': 0.9098, 'learning_rate': 0.000999046843662503, 'epoch': 0.01}                                                     
{'loss': 0.5898, 'learning_rate': 0.0009970611012927184, 'epoch': 0.01}                                                    
{'loss': 0.5583, 'learning_rate': 0.0009950753589229333, 'epoch': 0.02}                                                  
{'loss': 0.5469, 'learning_rate': 0.0009930896165531485, 'epoch': 0.02}                                          
{'loss': 0.5959, 'learning_rate': 0.0009911038741833634, 'epoch': 0.03}
```

## 合并模型

微调完成之后会有两个模型，第一个是Whisper基础模型，第二个是Lora模型，需要把这两个模型合并之后才能之后的操作。这个程序只需要传递两个参数，`--lora_model`指定的是训练时保存的检查点路径，注意后面还有`adapter_model`，第二个`--output_dir`是合并后模型的保存目录。
```shell
python merge_lora.py --lora_model=output/checkpoint-final --output_dir=models/
```

## 评估模型

执行以下程序进行评估模型，最重要的两个参数分别是。第一个`--model_path`指定的是合并后的模型路径，同时也支持直接使用Whisper原模型，例如直接指定`openai/whisper-large-v2`，第二个是`--metric`指定的是评估方法，例如有字错率`cer`和词错率`wer`。**提示：** 没有微调的模型，可能输出带有标点符号，影响准确率。其他更多的参数请查看这个程序。
```shell
python evaluation.py --model_path=models/whisper-tiny-finetune --metric=cer
```

以下是使用AIShell的微调前和微调后的字错率对比，使用whisper-tiny最为明显，它准确率比较低，但是微调之后有大幅度提升。

|      模型      |   微调前   |   微调后   |
|:------------:|:-------:|:-------:|
| whisper-tiny | 0.48265 | 0.17926 |


## 预测

执行以下程序进行语音识别，第一个`--audio_path`参数指定的是要预测的音频路径。第二个`--model_path`指定的是合并后的模型路径，同时也支持直接使用Whisper原模型，例如直接指定`openai/whisper-large-v2`。其他更多的参数请查看这个程序。
```shell
python infer.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune
```

# 加速预测

众所周知，直接使用Whisper模型推理是比较慢的，所以这里提供了一个加速的方式，主要是使用了CTranslate2进行加速，首先要转换模型，把合并后的模型转换为CTranslate2模型。如下命令，`--model`参数指定的是合并后的模型路径，同时也支持直接使用Whisper原模型，例如直接指定`openai/whisper-large-v2`。`--output_dir`参数指定的是转换后的CTranslate2模型路径，`--quantization`参数指定的是量化模型大小，不希望量化模型的可以直接去掉这个参数。
```shell
ct2-transformers-converter --model models/whisper-tiny-finetune --output_dir models/whisper-tiny-ct2 --copy_files tokenizer.json --quantization float16
```

执行以下程序进行加速语音识别，`--audio_path`参数指定的是要预测的音频路径。`--model_path`指定的是转换后的CTranslate2模型。其他更多的参数请查看这个程序。
```shell
python infer_ct2.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-ct2
```

输出结果如下：
```shell
{
    "language": "zh",
    "duration": 8.39,
    "results": [
        {
            "start": 0.0,
            "end": 8.39,
            "text": "近几年不但我用书给女儿压岁也劝说亲朋友不要给女儿压岁钱而改送压岁书"
        }
    ],
    "text": "近几年不但我用书给女儿压岁也劝说亲朋友不要给女儿压岁钱而改送压岁书"
}
```


