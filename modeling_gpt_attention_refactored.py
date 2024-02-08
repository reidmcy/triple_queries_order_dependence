import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Model,GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import types
from transformers.generation import utils as generation_utils
from transformers.utils import ModelOutput
import json
VERBOSE=False

def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # Note: this is where the 2D causal mask is generated, we override this to replace the causual mask with attention_mask_2D, if it is given
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            
            # Causal_mask has shape (1, 1, query_length, key_length)
            if attention_mask is not None and attention_mask.shape[-1] == causal_mask.shape[-1]**2:
                causal_mask = attention_mask.view(causal_mask.shape)#.to(causal_mask.dtype)
                attn_weights = attn_weights + causal_mask
            else:
                # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
                # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
                attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
            
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    

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
                    # given an attention_mask of shape (bsz, 1, tgt_seq_len, src_seq_len) in model kwargs
                    # TODO: the new attention_mask should reflect the padding strategy used in the original input sequence for the original attention mask
                    model_kwargs["attention_mask"] = torch.ones([attention_mask.shape[0], attention_mask.shape[-1]+1], dtype=torch.int32)
                    # next_position_id = model_kwargs["position_ids"][0][-1]+1 if "position_ids" in model_kwargs else attention_mask.shape[3]
                    # TODO: is next position id definition correct?
                    #next_position_id = attention_mask.shape[3]
                    #model_kwargs["position_ids"] = torch.tensor([[next_position_id]])
                else:
                    assert(len(attention_mask.shape) == 2)
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )
        return model_kwargs

def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        elif attention_mask is not None and position_ids is not None and len(attention_mask.shape)==4:
            # Don't set position_ids = None if the user passed in both position_ids and a 2D attention_mask
            assert(position_ids.shape[-1] == attention_mask.shape[-1])
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

def get_2D_attention_accepting_model_gpt(model):
    model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, model)
    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model)
    for hidden_layer in range(len(modelGPT.transformer.h)):
        model.transformer.h[hidden_layer].attn._attn = types.MethodType(_attn, model.transformer.h[hidden_layer].attn)
    return model

'''
modelGPT=transformers.GPT2LMHeadModel.from_pretrained("gpt2")
modelGPT=get_attention_accepting_model(modelGPT)
model=modelGPT

prompt=prompts_two_options[10]
seqA,seqB,seqC,seqD=prompt[0],prompt[1][0],prompt[1][1],prompt[2]
model,tokenizer=modelGPT,GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gen=model.generate(tok['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask=tokAll["attention_mask"], position_ids=position_ids, return_dict_in_generate=True, output_scores=True)

# position_ids=torch.arange(24).unsqueeze(0)
# TODO: position_ids are not being used?? gen is identical to genA (as measured by scores_diff), despite the position_ids being different
gen=model.generate(tokAll['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask=attention_mask_2d, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
gen2=model.generate(tokAll['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask=None, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
gen3=model.generate(tokRev['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask=attention_mask_2dRev, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
gen4=model.generate(tokRev['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask=None, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)

print(tokenizer.decode(gen.sequences[0], skip_special_tokens=True))
print(tokenizer.decode(gen2.sequences[0], skip_special_tokens=True))

print(tokenizer.decode(gen3.sequences[0], skip_special_tokens=True))
print(tokenizer.decode(gen4.sequences[0], skip_special_tokens=True))

gen5 = modelGPTOld.generate(tokAll['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask_2d=attention_mask_2d, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
gen6=modelGPTOld.generate(tokAll['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask_2d=None, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
gen7=modelGPTOld.generate(tokRev['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask_2d=attention_mask_2dRev, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
gen8=modelGPTOld.generate(tokRev['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask_2d=None, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)

gen9=modelGPTOld.generate(tokAll['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask_2d=attention_mask_2d, position_ids=torch.arange(24).unsqueeze(0), return_dict_in_generate=True, output_scores=True)
genA=model.generate(tokAll['input_ids'], max_new_tokens=MAX_NEW_TOKENS, attention_mask=attention_mask_2d, position_ids=torch.arange(26).unsqueeze(0), return_dict_in_generate=True, output_scores=True)
assert(scores_diff(gen5,gen9).item()!=0.0) # should be true that specifying position_ids changes the model output. Old model
assert(scores_diff(gen,genA).item()!=0.0) # should be true that specifying position_ids changes the model output. New model

print(scores_diff(gen, gen5))
print(scores_diff(gen2,gen6))
print(scores_diff(gen3, gen7))
print(scores_diff(gen4,gen8))

print(scores_diff(gen,gen3)) # with attention mask
print(scores_diff(gen2,gen4)) # without attention mask
print(scores_diff(gen5,gen7)) # with attention mask
print(scores_diff(gen6,gen8)) # without attention mask

print(scores_diff(gen,gen9))

#gen=modelGPT.generate(tokAll['input_ids'],position_ids=position_ids,attention_mask=attention_mask_2d,max_new_tokens=10,return_dict_in_generate=True)

outputs_two_options = []
diffs_two_options = []
for prompt in prompts_two_options[10:20]:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev, diff = testInterventionsRefactored(prompt[0],prompt[1][0],prompt[1][1],prompt[2],model=modelLlama,tokenizer=tokenizerLlama)
    outputs_two_options.append([textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt])
    diffs_two_options.append(diff)
df = pd.DataFrame.from_records(diffs_two_options).astype(float)
# save outputs_two_options array as json
with open('outputs_two_options_llama.json', 'w') as f:
    json.dump(outputs_two_options, f)
print(df)


# gen=modelLlama.generate(input_ids=tokAll['input_ids'], max_new_tokens=1, attention_mask=attention_mask_2d, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
'''