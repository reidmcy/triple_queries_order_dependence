import torch
from transformers import LlamaModel
from transformers.models.llama.modeling_llama import _make_causal_mask, LlamaPreTrainedModel
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import types
from transformers.generation import utils as generation_utils
from transformers.utils import ModelOutput
VERBOSE=False

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape[0],mask.shape[-1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    if mask.shape==(bsz,1,tgt_len,src_len):
        # If a 2D attention mask is passed in, use it as given
        expanded_mask = mask.to(dtype)
    else:
        # If a 1D attention mask is passed in, expand it to the 2D attention mask following the default huggingface behavior
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
            # replace -inf values with torch.finfo(dtype).min in combined_attention_mask
            combined_attention_mask = combined_attention_mask.masked_fill(combined_attention_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        return combined_attention_mask

def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                if len(attention_mask.shape) == 4:
                    # given an attention_mask of shape (bsz, 1, tgt_seq_len, src_seq_len) in model kwargs, generate 
                    # new 2D attention_mask of ones with shape (bsz, src_seq_len+1)
                    # TODO: this ignores any padding tokens in the input_ids, which may be a problem?!?!
                    model_kwargs["attention_mask"] = torch.ones([attention_mask.shape[0], attention_mask.shape[-1]+1])
                    next_position_id = model_kwargs["position_ids"][0][-1]+1 if "position_ids" in model_kwargs else attention_mask.shape[-1]
                    model_kwargs["position_ids"] = torch.tensor([[next_position_id]])
                else:
                    # Extend the length of the attention mask by 1 to reflect the fact that we have generated one additional token
                    assert(len(attention_mask.shape) == 2)
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                    next_position_id = model_kwargs["position_ids"][0][-1]+1 if "position_ids" in model_kwargs else attention_mask.shape[-1]
                    model_kwargs["position_ids"] = torch.tensor([[next_position_id]])
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

def get_2D_attention_accepting_model_llama(model):
    model.model._prepare_decoder_attention_mask = types.MethodType(_prepare_decoder_attention_mask, model.model)
    model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, model)
    return model
# modelLlama=get_attention_accepting_model(modelLlama)
