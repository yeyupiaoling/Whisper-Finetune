import json
import os
import re

import numpy as np
import paddle.inference as paddle_infer
from paddlenlp.transformers import ErnieTokenizer


__all__ = ['PunctuationExecutor']


class PunctuationExecutor:
    def __init__(self, model_dir, use_gpu=True, gpu_mem=500, num_threads=4):
        # 创建 config
        model_path = os.path.join(model_dir, 'model.pdmodel')
        params_path = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise Exception("标点符号模型文件不存在，请检查{}和{}是否存在！".format(model_path, params_path))
        self.config = paddle_infer.Config(model_path, params_path)
        # 获取预训练模型类型
        pretrained_token = 'ernie-1.0'
        if os.path.exists(os.path.join(model_dir, 'info.json')):
            with open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                pretrained_token = data['pretrained_token']

        if use_gpu:
            self.config.enable_use_gpu(gpu_mem, 0)
        else:
            self.config.disable_gpu()
            self.config.set_cpu_math_library_num_threads(num_threads)
        # enable memory optim
        self.config.enable_memory_optim()
        self.config.disable_glog_info()

        # 根据 config 创建 predictor
        self.predictor = paddle_infer.create_predictor(self.config)

        # 获取输入层
        self.input_ids_handle = self.predictor.get_input_handle('input_ids')
        self.token_type_ids_handle = self.predictor.get_input_handle('token_type_ids')

        # 获取输出的名称
        self.output_names = self.predictor.get_output_names()

        self._punc_list = []
        if not os.path.join(model_dir, 'vocab.txt'):
            raise Exception("字典文件不存在，请检查{}是否存在！".format(os.path.join(model_dir, 'vocab.txt')))
        with open(os.path.join(model_dir, 'vocab.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                self._punc_list.append(line.strip())

        self.tokenizer = ErnieTokenizer.from_pretrained(pretrained_token)

        # 预热
        self('近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书')

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
        text = re.sub(f'[{"".join([p for p in self._punc_list][1:])}]', '', text)
        return text

    # 预处理文本
    def preprocess(self, text: str):
        clean_text = self._clean_text(text)
        if len(clean_text) == 0: return None
        tokenized_input = self.tokenizer(list(clean_text), return_length=True, is_split_into_words=True)
        input_ids = tokenized_input['input_ids']
        seg_ids = tokenized_input['token_type_ids']
        seq_len = tokenized_input['seq_len']
        return input_ids, seg_ids, seq_len

    def infer(self, input_ids: list, seg_ids: list):
        # 设置输入
        self.input_ids_handle.reshape([1, len(input_ids)])
        self.token_type_ids_handle.reshape([1, len(seg_ids)])
        self.input_ids_handle.copy_from_cpu(np.array([input_ids]).astype('int64'))
        self.token_type_ids_handle.copy_from_cpu(np.array([seg_ids]).astype('int64'))

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_data = output_handle.copy_to_cpu()
        return output_data

    # 后处理识别结果
    def postprocess(self, input_ids, seq_len, preds):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:seq_len - 1])
        labels = preds[1:seq_len - 1].tolist()
        assert len(tokens) == len(labels)

        text = ''
        for t, l in zip(tokens, labels):
            text += t
            if l != 0:
                text += self._punc_list[l]
        return text

    def __call__(self, text: str) -> str:
        # 数据batch处理
        input_ids, seg_ids, seq_len = self.preprocess(text)
        preds = self.infer(input_ids=input_ids, seg_ids=seg_ids)
        if len(preds.shape) == 2:
            preds = preds[0]
        text = self.postprocess(input_ids, seq_len, preds)
        return text