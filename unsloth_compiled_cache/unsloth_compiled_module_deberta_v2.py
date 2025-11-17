"""
2025.11.4
2025.11.3
4.57.1
0.23.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass


from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 32, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.deberta_v2.modeling_deberta_v2 import (Optional, torch, nn, LayerNorm, ACT2FN, DebertaV2Config, make_log_bucket_position, scaled_size_sqrt, build_rpos)

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DebertaV2SelfOutput_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        return DebertaV2SelfOutput_forward(self, hidden_states, input_tensor)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position
        device (`torch.device`): the device on which tensors will be created.

    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)

    q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.compiler.disable(recursive = False)
def DisentangledSelfAttention_forward(
    self,
    hidden_states,
    attention_mask,
    output_attentions=False,
    query_states=None,
    relative_pos=None,
    rel_embeddings=None,
):
    """
    Call the module

    Args:
        hidden_states (`torch.FloatTensor`):
            Input states to the module usually the output from previous layer, it will be the Q,K and V in
            *Attention(Q,K,V)*

        attention_mask (`torch.BoolTensor`):
            An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
            sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
            th token.

        output_attentions (`bool`, *optional*):
            Whether return the attention matrix.

        query_states (`torch.FloatTensor`, *optional*):
            The *Q* state in *Attention(Q,K,V)*.

        relative_pos (`torch.LongTensor`):
            The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
            values ranging in [*-max_relative_positions*, *max_relative_positions*].

        rel_embeddings (`torch.FloatTensor`):
            The embedding of relative distances. It's a tensor of shape [\\(2 \\times
            \\text{max_relative_positions}\\), *hidden_size*].


    """
    if query_states is None:
        query_states = hidden_states
    query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

    rel_att = None
    # Take the dot product between "query" and "key" to get the raw attention scores.
    scale_factor = 1
    if "c2p" in self.pos_att_type:
        scale_factor += 1
    if "p2c" in self.pos_att_type:
        scale_factor += 1
    scale = scaled_size_sqrt(query_layer, scale_factor)
    attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    if self.relative_attention:
        rel_embeddings = self.pos_dropout(rel_embeddings)
        rel_att = self.disentangled_attention_bias(
            query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
        )

    if rel_att is not None:
        attention_scores = attention_scores + rel_att
    attention_scores = attention_scores.view(
        -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
    )

    attention_mask = attention_mask.bool()
    attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
    # bsz x height x length x dimension
    attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype = torch.float32).to(attention_scores.dtype).to(attention_scores.dtype)

    attention_probs = self.dropout(attention_probs)
    context_layer = torch.bmm(
        attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    )
    context_layer = (
        context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    new_context_layer_shape = context_layer.size()[:-2] + (-1,)
    context_layer = context_layer.view(new_context_layer_shape)
    if not output_attentions:
        return (context_layer, None)
    return (context_layer, attention_probs)

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        return DisentangledSelfAttention_forward(self, hidden_states, attention_mask, output_attentions, query_states, relative_pos, rel_embeddings)

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer,
                key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(
                query_layer.size(0) // self.num_attention_heads, 1, 1
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_key_layer, scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]),
            )
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_query_layer, scale_factor)
            r_pos = build_rpos(
                query_layer,
                key_layer,
                relative_pos,
                self.max_relative_positions,
                self.position_buckets,
            )
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
            ).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DebertaV2Attention_forward(
    self,
    hidden_states,
    attention_mask,
    output_attentions: bool = False,
    query_states=None,
    relative_pos=None,
    rel_embeddings=None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    self_output, att_matrix = self.self(
        hidden_states,
        attention_mask,
        output_attentions,
        query_states=query_states,
        relative_pos=relative_pos,
        rel_embeddings=rel_embeddings,
    )
    if query_states is None:
        query_states = hidden_states
    attention_output = self.output(self_output, query_states)

    if output_attentions:
        return (attention_output, att_matrix)
    else:
        return (attention_output, None)

class DebertaV2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return DebertaV2Attention_forward(self, hidden_states, attention_mask, output_attentions, query_states, relative_pos, rel_embeddings)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DebertaV2Intermediate_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states

class DebertaV2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return DebertaV2Intermediate_forward(self, hidden_states)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DebertaV2Output_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class DebertaV2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        return DebertaV2Output_forward(self, hidden_states, input_tensor)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def ConvLayer_forward(self, hidden_states, residual_states, input_mask):
    out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
    rmask = (1 - input_mask).bool()
    out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
    out = ACT2FN[self.conv_act](self.dropout(out))

    layer_norm_input = residual_states + out
    output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

    if input_mask is None:
        output_states = output
    else:
        if input_mask.dim() != layer_norm_input.dim():
            if input_mask.dim() == 4:
                input_mask = input_mask.squeeze(1).squeeze(1)
            input_mask = input_mask.unsqueeze(2)

        input_mask = input_mask.to(output.dtype)
        output_states = output * input_mask

    return output_states

class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        return ConvLayer_forward(self, hidden_states, residual_states, input_mask)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def LegacyDebertaV2LMPredictionHead_forward(self, hidden_states):
    hidden_states = self.transform(hidden_states)
    hidden_states = self.decoder(hidden_states)
    return hidden_states

class LegacyDebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LegacyDebertaV2PredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        return LegacyDebertaV2LMPredictionHead_forward(self, hidden_states)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DebertaV2LMPredictionHead_forward(self, hidden_states, word_embeddings):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.LayerNorm(hidden_states)
    hidden_states = torch.matmul(hidden_states, word_embeddings.weight.t()) + self.bias
    return hidden_states

class DebertaV2LMPredictionHead(nn.Module):
    """https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py#L270"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    # note that the input embeddings must be passed as an argument
    def forward(self, hidden_states, word_embeddings):
        return DebertaV2LMPredictionHead_forward(self, hidden_states, word_embeddings)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def LegacyDebertaV2PredictionHeadTransform_forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.LayerNorm(hidden_states)
    return hidden_states

class LegacyDebertaV2PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        return LegacyDebertaV2PredictionHeadTransform_forward(self, hidden_states)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def LegacyDebertaV2OnlyMLMHead_forward(self, sequence_output):
    prediction_scores = self.predictions(sequence_output)
    return prediction_scores

class LegacyDebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LegacyDebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        return LegacyDebertaV2OnlyMLMHead_forward(self, sequence_output)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DebertaV2OnlyMLMHead_forward(self, sequence_output, word_embeddings):
    prediction_scores = self.lm_head(sequence_output, word_embeddings)
    return prediction_scores

class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = DebertaV2LMPredictionHead(config)

    # note that the input embeddings must be passed as an argument
    def forward(self, sequence_output, word_embeddings):
        return DebertaV2OnlyMLMHead_forward(self, sequence_output, word_embeddings)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def ContextPooler_forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.

    context_token = hidden_states[:, 0]
    context_token = self.dropout(context_token)
    pooled_output = self.dense(context_token)
    pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
    return pooled_output

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = nn.Dropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        return ContextPooler_forward(self, hidden_states)

    @property
    def output_dim(self):
        return self.config.hidden_size
