import argparse
import functools
import json
import os
import struct

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration

from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_dir",   type=str,  default="models/whisper-tiny-finetune", help="需要转换的模型路径")
add_arg("output_path", type=str,  default="models/ggml-model.bin", help="转换保存模型的路径")
add_arg("use_f16",     type=bool, default=True,                    help="是否量化为半精度")
args = parser.parse_args()
print_arguments(args)

conv_map = {
    'self_attn.k_proj': 'attn.key',
    'self_attn.q_proj': 'attn.query',
    'self_attn.v_proj': 'attn.value',
    'self_attn.out_proj': 'attn.out',
    'self_attn_layer_norm': 'attn_ln',
    'encoder_attn.q_proj': 'cross_attn.query',
    'encoder_attn.v_proj': 'cross_attn.value',
    'encoder_attn.out_proj': 'cross_attn.out',
    'encoder_attn_layer_norm': 'cross_attn_ln',
    'fc1': 'mlp.0',
    'fc2': 'mlp.2',
    'final_layer_norm': 'mlp_ln',
    'encoder.layer_norm.bias': 'encoder.ln_post.bias',
    'encoder.layer_norm.weight': 'encoder.ln_post.weight',
    'encoder.embed_positions.weight': 'encoder.positional_embedding',
    'decoder.layer_norm.bias': 'decoder.ln.bias',
    'decoder.layer_norm.weight': 'decoder.ln.weight',
    'decoder.embed_positions.weight': 'decoder.positional_embedding',
    'decoder.embed_tokens.weight': 'decoder.token_embedding.weight',
    'proj_out.weight': 'decoder.proj.weight',
}


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


encoder = json.load(open(f"{args.model_dir}/vocab.json", "r", encoding="utf8"))
encoder_added = json.load(open(f"{args.model_dir}/added_tokens.json", "r", encoding="utf8"))
hparams = json.load(open(f"{args.model_dir}/config.json", "r", encoding="utf8"))
# 支持large-v3模型
if "max_length" not in hparams.keys():
    hparams["max_length"] = hparams["max_target_positions"]

model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)

n_mels = hparams["num_mel_bins"]
with np.load(f"tools/mel_filters.npz") as f:
    filters = torch.from_numpy(f[f"mel_{n_mels}"])

tokens = json.load(open(f"{args.model_dir}/vocab.json", "r", encoding="utf8"))

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
fout = open(args.output_path, "wb")

fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_source_positions"]))
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["encoder_attention_heads"]))
fout.write(struct.pack("i", hparams["encoder_layers"]))
fout.write(struct.pack("i", hparams["max_length"] if hparams["max_length"] else 448))
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["decoder_attention_heads"]))
fout.write(struct.pack("i", hparams["decoder_layers"]))
fout.write(struct.pack("i", hparams["num_mel_bins"]))
fout.write(struct.pack("i", args.use_f16))

fout.write(struct.pack("i", filters.shape[0]))
fout.write(struct.pack("i", filters.shape[1]))
for i in range(filters.shape[0]):
    for j in range(filters.shape[1]):
        fout.write(struct.pack("f", filters[i][j]))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(tokens)))

tokens = sorted(tokens.items(), key=lambda x: x[1])
for key in tokens:
    text = bytearray([byte_decoder[c] for c in key[0]])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

list_vars = model.state_dict()
for name in list_vars.keys():
    # this seems to not be used
    if name == "proj_out.weight":
        print('Skipping', name)
        continue

    src = name

    nn = name
    if name != "proj_out.weight":
        nn = nn.split(".")[1:]
    else:
        nn = nn.split(".")

    if nn[1] == "layers":
        nn[1] = "blocks"
        if ".".join(nn[3:-1]) == "encoder_attn.k_proj":
            mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
        else:
            mapped = conv_map[".".join(nn[3:-1])]
        name = ".".join(nn[:3] + [mapped] + nn[-1:])
    else:
        name = ".".join(nn)
        name = conv_map[name] if name in conv_map else name

    print(src, ' -> ', name)
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float16)

    # reshape conv bias from [n] to [n, 1]
    if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
        data = data.reshape(data.shape[0], 1)
        print("  Reshaped variable: ", name, " to shape: ", data.shape)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # looks like the whisper models are in f16 by default
    # so we need to convert the small tensors to f32 until we fully support f16 in ggml
    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 1
    if args.use_f16:
        if n_dims < 2 or \
                name == "encoder.conv1.bias" or \
                name == "encoder.conv2.bias" or \
                name == "encoder.positional_embedding" or \
                name == "decoder.positional_embedding":
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0
    else:
        data = data.astype(np.float32)
        ftype = 0

    # header
    str_ = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str_)

    # data
    data.tofile(fout)

fout.close()

print(f"导出模型: {args.output_path}")
