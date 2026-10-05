"""Log-Mel encoder/decoder Transformer for ASR with Hugging Face Trainer support.

Inputs use the Whisper feature layout: ``input_features[B, num_mel_bins, frames]``.
``attention_mask[B, frames]`` uses 1 for audio and 0 for right padding. Labels
are token IDs with padding replaced by -100; do not prepend the decoder start
token. Set vocabulary size and special token IDs to match the chosen tokenizer.

This model is trained from scratch, not a drop-in loader for Whisper weights.
Generation recomputes the decoder prefix (no KV cache). Import this module before
loading its checkpoints through AutoConfig / AutoModelForSpeechSeq2Seq.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class ASRTransformerConfig(PretrainedConfig):
	model_type = "asr_transformer"
	attribute_map = {
		"hidden_size": "d_model",
		"num_attention_heads": "encoder_attention_heads",
		"num_hidden_layers": "encoder_layers",
	}

	def __init__(
		self,
		vocab_size=51865,
		num_mel_bins=80,
		d_model=256,
		encoder_layers=6,
		decoder_layers=4,
		encoder_attention_heads=4,
		decoder_attention_heads=4,
		encoder_ffn_dim=1024,
		decoder_ffn_dim=1024,
		dropout=0.1,
		max_source_positions=1500,
		max_target_positions=448,
		initializer_range=0.02,
		pad_token_id=0,
		bos_token_id=1,
		eos_token_id=2,
		decoder_start_token_id=1,
		**kwargs,
	):
		# Both values are fixed for this architecture, including on reload.
		kwargs.pop("is_encoder_decoder", None)
		kwargs.pop("use_cache", None)
		tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
		if tie_word_embeddings:
			raise ValueError("ASRTransformer uses separate input and output embeddings")
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			decoder_start_token_id=decoder_start_token_id,
			is_encoder_decoder=True,
			tie_word_embeddings=False,
			**kwargs,
		)
		for name, value in {
			"vocab_size": vocab_size, "num_mel_bins": num_mel_bins, "d_model": d_model,
			"encoder_layers": encoder_layers, "decoder_layers": decoder_layers,
			"encoder_attention_heads": encoder_attention_heads,
			"decoder_attention_heads": decoder_attention_heads,
			"encoder_ffn_dim": encoder_ffn_dim, "decoder_ffn_dim": decoder_ffn_dim,
			"max_source_positions": max_source_positions,
			"max_target_positions": max_target_positions,
		}.items():
			if not isinstance(value, int) or value <= 0:
				raise ValueError(f"{name} must be a positive integer")
			setattr(self, name, value)
		if d_model % encoder_attention_heads or d_model % decoder_attention_heads:
			raise ValueError("d_model must be divisible by both attention head counts")
		if not 0 <= dropout < 1:
			raise ValueError("dropout must be in [0, 1)")
		for name in ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id"):
			value = getattr(self, name)
			if not isinstance(value, int) or not 0 <= value < vocab_size:
				raise ValueError(f"{name} must be an integer in [0, vocab_size)")
		self.dropout = dropout
		self.initializer_range = initializer_range
		self.use_cache = False


class SinusoidalPositionEmbedding(nn.Module):
	def __init__(self, d_model, max_positions):
		super().__init__()
		position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
		frequency = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		embedding = torch.zeros(max_positions, d_model)
		embedding[:, 0::2] = torch.sin(position * frequency)
		embedding[:, 1::2] = torch.cos(position * frequency[:d_model // 2])
		# Persist this buffer: HF's meta-device loading otherwise leaves a
		# non-persistent sinusoidal table uninitialized after from_pretrained.
		self.register_buffer("embedding", embedding)

	def forward(self, hidden_states):
		length = hidden_states.shape[1]
		if length > self.embedding.shape[0]:
			raise ValueError(f"Sequence length {length} exceeds positional limit {self.embedding.shape[0]}")
		return hidden_states + self.embedding[:length].to(dtype=hidden_states.dtype)

class RotaryPositionEmbedding(nn.Module):
	def __init__(d_model, max_positions):
		position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
		frequency = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
		

def _audio_mask(attention_mask, batch_size, frames, device):
	if attention_mask is None:
		return torch.ones(batch_size, frames, dtype=torch.bool, device=device)
	if attention_mask.shape != (batch_size, frames):
		raise ValueError("attention_mask must have shape [batch, input audio frames]")
	mask = attention_mask.to(device=device, dtype=torch.bool)
	if not mask.any(dim=1).all():
		raise ValueError("Each audio example must contain at least one valid frame")
	if (mask[:, 1:] & ~mask[:, :-1]).any():
		raise ValueError("Audio attention_mask must use right padding")
	return mask


def _subsample_mask(mask):
	# kernel=3, stride=2, padding=1 gives ceil(length / 2).
	lengths = (mask.sum(dim=1) + 1) // 2
	positions = torch.arange((mask.shape[1] + 1) // 2, device=mask.device)
	return positions.unsqueeze(0) < lengths.unsqueeze(1)


class ASRTransformerEncoder(nn.Module):
	main_input_name = "input_features"

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, stride=2, padding=1)
		self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
		self.positions = SinusoidalPositionEmbedding(config.d_model, config.max_source_positions)
		self.dropout = nn.Dropout(config.dropout)
		self.layers = nn.ModuleList([
			nn.TransformerEncoderLayer(
				config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim,
				config.dropout, activation="gelu", batch_first=True, norm_first=True,
			) for _ in range(config.encoder_layers)
		])
		self.layer_norm = nn.LayerNorm(config.d_model)

	def forward(self, input_features, attention_mask=None, output_hidden_states=False,
				output_attentions=False, return_dict=True, **kwargs):
		if output_attentions:
			raise ValueError("PyTorch Transformer layers do not expose attention weights")
		if input_features.ndim != 3 or input_features.shape[1] != self.config.num_mel_bins:
			raise ValueError("input_features must have shape [batch, num_mel_bins, frames]")
		if input_features.shape[2] == 0:
			raise ValueError("Input audio must not be empty")
		if input_features.shape[2] > 4 * self.config.max_source_positions:
			raise ValueError("Input audio exceeds 4 * max_source_positions frames; split it into chunks")
		mask = _audio_mask(attention_mask, input_features.shape[0], input_features.shape[2], input_features.device)
		hidden = input_features.masked_fill(~mask.unsqueeze(1), 0)
		# Zero padding between convolutions too, so padded values cannot leak
		# into the last valid feature through the second convolution.
		for conv in (self.conv1, self.conv2):
			hidden = F.gelu(conv(hidden))
			mask = _subsample_mask(mask)
			hidden = hidden.masked_fill(~mask.unsqueeze(1), 0)
		hidden = self.dropout(self.positions(hidden.transpose(1, 2)))
		states = () if output_hidden_states else None
		for layer in self.layers:
			if output_hidden_states:
				states += (hidden,)
			hidden = layer(hidden, src_key_padding_mask=~mask)
		hidden = self.layer_norm(hidden).masked_fill(~mask.unsqueeze(-1), 0)
		if output_hidden_states:
			states += (hidden,)
		output = BaseModelOutput(last_hidden_state=hidden, hidden_states=states)
		return output if return_dict else output.to_tuple()


class ASRTransformerForConditionalGeneration(PreTrainedModel, GenerationMixin):
	config_class = ASRTransformerConfig
	main_input_name = "input_features"
	base_model_prefix = "asr_transformer"
	# Loss is already averaged over valid tokens, not Trainer's item count.
	accepts_loss_kwargs = False

	def __init__(self, config):
		super().__init__(config)
		self.encoder = ASRTransformerEncoder(config)
		self.decoder_embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
		self.decoder_positions = SinusoidalPositionEmbedding(config.d_model, config.max_target_positions)
		self.decoder_dropout = nn.Dropout(config.dropout)
		self.decoder_layers = nn.ModuleList([
			nn.TransformerDecoderLayer(
				config.d_model, config.decoder_attention_heads, config.decoder_ffn_dim,
				config.dropout, activation="gelu", batch_first=True, norm_first=True,
			) for _ in range(config.decoder_layers)
		])
		self.decoder_layer_norm = nn.LayerNorm(config.d_model)
		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
		self.post_init()

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
			nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
			if getattr(module, "bias", None) is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.MultiheadAttention):
			if module.in_proj_weight is not None:
				nn.init.normal_(module.in_proj_weight, mean=0.0, std=self.config.initializer_range)
			if module.in_proj_bias is not None:
				nn.init.zeros_(module.in_proj_bias)
		elif isinstance(module, nn.LayerNorm):
			nn.init.ones_(module.weight)
			nn.init.zeros_(module.bias)

	def get_encoder(self):
		return self.encoder

	def get_input_embeddings(self):
		return self.decoder_embed_tokens

	def set_input_embeddings(self, value):
		self.decoder_embed_tokens = value

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, value):
		self.lm_head = value

	def prepare_decoder_input_ids_from_labels(self, labels):
		if labels.ndim != 2 or labels.shape[1] == 0:
			raise ValueError("labels must have shape [batch, nonempty target length]")
		shifted = labels.new_full(labels.shape, self.config.pad_token_id)
		shifted[:, 0] = self.config.decoder_start_token_id
		shifted[:, 1:] = labels[:, :-1]
		return shifted.masked_fill(shifted == -100, self.config.pad_token_id)

	@classmethod
	def _supports_default_dynamic_cache(cls):
		return False

	def prepare_inputs_for_generation(self, decoder_input_ids, attention_mask=None,
									  encoder_outputs=None, decoder_attention_mask=None, **kwargs):
		# Keep the whole prefix: nn.TransformerDecoderLayer has no KV cache API.
		return {
			"decoder_input_ids": decoder_input_ids,
			"decoder_attention_mask": decoder_attention_mask,
			"encoder_outputs": encoder_outputs,
			"attention_mask": attention_mask,
			"use_cache": False,
		}

	def forward(
		self,
		input_features: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.Tensor] = None,
		decoder_attention_mask: Optional[torch.Tensor] = None,
		encoder_outputs=None,
		labels: Optional[torch.Tensor] = None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		**kwargs,
	):
		return_dict = self.config.return_dict if return_dict is None else return_dict
		output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
		output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
		if output_attentions:
			raise ValueError("PyTorch Transformer layers do not expose attention weights")
		if encoder_outputs is None:
			if input_features is None:
				raise ValueError("Provide input_features or precomputed encoder_outputs")
			encoder_outputs = self.encoder(
				input_features, attention_mask=attention_mask,
				output_hidden_states=output_hidden_states, return_dict=True,
			)
		elif isinstance(encoder_outputs, tuple):
			encoder_outputs = BaseModelOutput(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
			)
		memory = encoder_outputs.last_hidden_state
		memory_padding_mask = None
		if attention_mask is not None:
			mask = _audio_mask(attention_mask, memory.shape[0], attention_mask.shape[1], memory.device)
			mask = _subsample_mask(_subsample_mask(mask))
			if mask.shape[1] != memory.shape[1]:
				raise ValueError("attention_mask must describe audio frames before convolution subsampling")
			memory_padding_mask = ~mask

		if decoder_input_ids is None:
			if labels is None:
				raise ValueError("Provide labels for training or decoder_input_ids for decoding")
			decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
		if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[1] == 0:
			raise ValueError("decoder_input_ids must have shape [batch, nonempty target length]")
		if decoder_attention_mask is None:
			if labels is not None:
				decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.bool)
				decoder_attention_mask[:, 1:] = labels[:, :-1].ne(-100)
			else:
				decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id)
				decoder_attention_mask[:, 0] = True  # decoder start may equal pad
		if decoder_attention_mask.shape != decoder_input_ids.shape:
			raise ValueError("decoder_attention_mask must match decoder_input_ids")
		decoder_padding_mask = ~decoder_attention_mask.to(device=decoder_input_ids.device, dtype=torch.bool)
		hidden = self.decoder_embed_tokens(decoder_input_ids) * math.sqrt(self.config.d_model)
		hidden = self.decoder_dropout(self.decoder_positions(hidden))
		causal_mask = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=hidden.device).triu(1)
		states = () if output_hidden_states else None
		for layer in self.decoder_layers:
			if output_hidden_states:
				states += (hidden,)
			hidden = layer(
				hidden, memory, tgt_mask=causal_mask,
				tgt_key_padding_mask=decoder_padding_mask,
				memory_key_padding_mask=memory_padding_mask,
			)
		hidden = self.decoder_layer_norm(hidden)
		if output_hidden_states:
			states += (hidden,)
		logits = self.lm_head(hidden)
		loss = None
		if labels is not None:
			if labels.shape != logits.shape[:2]:
				raise ValueError("labels and decoder_input_ids must have the same shape")
			# Avoid NaN when a batch contains only ignored targets.
			loss = (F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
									labels.reshape(-1), ignore_index=-100)
					if labels.ne(-100).any() else logits.sum() * 0.0)
		output = Seq2SeqLMOutput(
			loss=loss, logits=logits, decoder_hidden_states=states,
			encoder_last_hidden_state=memory,
			encoder_hidden_states=encoder_outputs.hidden_states,
		)
		return output if return_dict else output.to_tuple()


# Registration is local to this Python process; no remote code execution needed.
AutoConfig.register(ASRTransformerConfig.model_type, ASRTransformerConfig, exist_ok=True)
AutoModelForSpeechSeq2Seq.register(ASRTransformerConfig, ASRTransformerForConditionalGeneration, exist_ok=True)

# Short aliases for training scripts.
TransformerConfig = ASRTransformerConfig
TransformerForConditionalGeneration = ASRTransformerForConditionalGeneration

