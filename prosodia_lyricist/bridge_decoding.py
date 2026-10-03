"""Inference-only KV cache for the bridge's standard pre-norm PyTorch decoder.

Training/checkpoint parameters stay in nn.TransformerDecoder. This executes the
same layers one position at a time, caching source projections and target K/V.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _heads(x, count):
    return x.unflatten(-1, (count, x.shape[-1] // count)).transpose(1, 2)


def _attention(module, query, keys, values, keep):
    attended = F.scaled_dot_product_attention(
        _heads(query, module.num_heads), keys, values,
        attn_mask=keep[:, None, None, :], dropout_p=0.0,
    )
    return module.out_proj(attended.transpose(1, 2).flatten(-2))


class DecoderCache:
    def __init__(self, decoder, memory, source_mask, max_length):
        if decoder.training or any(not layer.norm_first for layer in decoder.layers):
            raise ValueError("Decoder caching requires an eval-mode pre-norm decoder")
        self.decoder, self.step_index = decoder, 0
        self.source_keep = source_mask.bool()
        self.target_keep = torch.zeros(
            len(memory), max_length, dtype=torch.bool, device=memory.device
        )
        self.layers = []
        for layer in decoder.layers:
            attention = layer.multihead_attn
            width = attention.embed_dim
            key, value = F.linear(
                memory, attention.in_proj_weight[width:], attention.in_proj_bias[width:]
            ).chunk(2, dim=-1)
            shape = (len(memory), layer.self_attn.num_heads, max_length,
                     width // layer.self_attn.num_heads)
            self.layers.append({
                "source_key": _heads(key, attention.num_heads),
                "source_value": _heads(value, attention.num_heads),
                "target_key": key.new_empty(shape),
                "target_value": value.new_empty(shape),
            })

    # The growing K/V prefix changes attention shapes at every token. cuDNN can
    # build and retain a separate host-side execution plan for each shape/stride.
    # Exclude that backend only during cached decoding; sdpa_kernel restores the
    # caller's backend settings on return or error. Math supports fallback masks.
    @sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    def step(self, x, active):
        position = self.step_index
        self.target_keep[:, position] = active
        for layer, cache in zip(self.decoder.layers, self.layers, strict=True):
            attention = layer.self_attn
            query, key, value = F.linear(
                layer.norm1(x), attention.in_proj_weight, attention.in_proj_bias
            ).chunk(3, dim=-1)
            cache["target_key"][:, :, position : position + 1] = _heads(key, attention.num_heads)
            cache["target_value"][:, :, position : position + 1] = _heads(
                value, attention.num_heads
            )
            x = x + _attention(
                attention, query, cache["target_key"][:, :, : position + 1],
                cache["target_value"][:, :, : position + 1],
                self.target_keep[:, : position + 1],
            )
            attention = layer.multihead_attn
            width = attention.embed_dim
            query = F.linear(
                layer.norm2(x), attention.in_proj_weight[:width], attention.in_proj_bias[:width]
            )
            x = x + _attention(
                attention, query, cache["source_key"], cache["source_value"], self.source_keep
            )
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        self.step_index += 1
        return self.decoder.norm(x) if self.decoder.norm is not None else x
