from attention_mask_editing import *
from modeling_gpt_attention_refactored import get_2D_attention_accepting_model_gpt
from modeling_llama_attention import get_2D_attention_accepting_model_llama
from transformers import GPT2LMHeadModel,LlamaForCausalLM

'''
genOrderIndpendentOutput() is the primary entryway for generating order independent output for a given huggingface model instance and text input sequence. 
'''

def get_2D_attention_accepting_model(model):
    if isinstance(model,GPT2LMHeadModel):
        return get_2D_attention_accepting_model_gpt(model)
    elif isinstance(model,LlamaForCausalLM):
        print(f"Modify llama model to accept 2D attention mask")
        return get_2D_attention_accepting_model_llama(model)
    else:
        raise ValueError(f"model_type={model.__class__} not recognized")
def get_tokenized_input_prompt(tokA,tokParallel,tokD):
    tokAll={'input_ids':tokA['input_ids'],'attention_mask':tokA['attention_mask']}
    for tokOption in tokParallel:
        tokAll['input_ids']=torch.cat((tokAll['input_ids'], tokOption['input_ids'][0].unsqueeze(0)), dim=1)
        tokAll['attention_mask']=torch.cat((tokAll['attention_mask'], tokOption['attention_mask'][0].unsqueeze(0)), dim=1)
    tokAll['input_ids']=torch.cat((tokAll['input_ids'], tokD['input_ids'][0].unsqueeze(0)), dim=1)
    tokAll['attention_mask']=torch.cat((tokAll['attention_mask'], tokD['attention_mask'][0].unsqueeze(0)), dim=1)
    return tokAll

def genOrderIndependentOutputSingleString(input_text, model, tokenizer, max_new_tokens=10, is_order_independent=True, reverse_parallel_substrings_order=False, torch_device="cpu"):
    # @param input_text: "A|B;C;D|E" where A,B,C,D,E are strings, and | and ; are delimiters. The number of substrings between | and | is arbitrary.
    # The substrings between | and |, delimited by ";", are processed in parallel by the model
    prefix,parallel_substrings,suffix = input_text.split("|")
    parallel_substrings=parallel_substrings.split(";")
    return genOrderIndependentOutput(prefix, parallel_substrings, suffix, model, tokenizer, max_new_tokens, is_order_independent, reverse_parallel_substrings_order, torch_device)

def genOrderIndependentOutput(prefix, parallel_substrings, suffix, model, tokenizer, max_new_tokens=10, is_order_independent=True, reverse_parallel_substrings_order=False, torch_device="cpu"):
    # Modify the given model to accept a 2D attention mask as input
    model = get_2D_attention_accepting_model(model)
    
    # Tokenize input text
    tokA=tokenizer(prefix, return_tensors='pt',add_special_tokens=True, return_token_type_ids=False).to(torch_device)
    tokD=tokenizer(suffix, return_tensors='pt',add_special_tokens=False, return_token_type_ids=False).to(torch_device)
    if reverse_parallel_substrings_order:
        parallel_substrings=parallel_substrings[::-1]
    tokParallel = tuple([tokenizer(input_text, return_tensors='pt', add_special_tokens=False).to(torch_device) for input_text in parallel_substrings])
    tokAll=get_tokenized_input_prompt(tokA,tokParallel,tokD)
    assert(len(tokA['attention_mask'][0]) + sum([len(tokOption['attention_mask'][0]) for tokOption in tokParallel]) + len(tokD['attention_mask'][0]) == len(tokAll['attention_mask'][0]))
    s=len(tokAll['input_ids'][0])
    inputTextLen=len(prefix)+sum([len(s) for s in parallel_substrings])+len(suffix)
    
    if not is_order_independent:
        # Run tests with no attention mask nor position_id intervention
        generated=model.generate(tokAll['input_ids'], max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True)
    else:
        # Pad all parallel substrings to the same length, then generate a 2D attention mask such that all substrings are processed in parallel
        position_ids, tokParallel, tokAll = get_position_ids_padded_n_options(tokA, tokParallel, tokD, tokenizer=tokenizer)
        attention_mask_2d = get_attention_mask_2d_n_options(tokA, tokParallel, tokD, tokAll)
        print(f"Gen text with attention mask with shape {attention_mask_2d.shape}")
        generated=model.generate(tokAll['input_ids'], max_new_tokens=max_new_tokens, attention_mask=attention_mask_2d, position_ids=position_ids, return_dict_in_generate=True, output_scores=True)
    text=tokenizer.decode(generated.sequences[0], skip_special_tokens=True)[inputTextLen:]
    # Return output of model generation, and generated text
    return generated,text

#prefix,parallel_substrings,suffix="You are a gremlin who is ",["kind,","grisly,"]," How would you greet someone?"
#g,t=genOrderIndependentOutput(prefix, parallel_substrings, suffix, modelGPT, tokenizerGPT)