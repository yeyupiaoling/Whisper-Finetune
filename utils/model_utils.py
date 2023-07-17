import bitsandbytes as bnb
import torch
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def find_all_linear_names(use_8bit, model):
    cls = bnb.nn.Linear8bitLt if use_8bit else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_modules = list(lora_module_names)
    return target_modules


def load_from_checkpoint(resume_from_checkpoint, model=None):
    pass
