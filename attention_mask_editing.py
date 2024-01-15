from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers import GPT2Model,GPT2LMHeadModel
import torch
import tensorflow as tf

import importlib
import transformers
import modeling_gpt_attention
importlib.reload(modeling_gpt_attention)
importlib.reload(transformers)
from modeling_gpt_attention import GPT2LMHeadModelAttention

'''
Attention mask:
- 1 for tokens that are **not masked**,
- 0 for tokens that are **masked**.


TODO: I think if I hack into the causual mask generation when it's called initially with a
    [1,1,8,8,] size causal mask, I can get it to work. Investigate! Figure out indexing, matrix isn't quite diagonal?
    What's the self.bias definition? 
'''

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Use GPT2 model that accepts 2D attention masks as input
model = GPT2LMHeadModelAttention.from_pretrained("gpt2", output_attentions=True)

def get_position_ids(tokA, tokB, tokC, tokD):
    nTokA = len(tokA['attention_mask'][0])
    nTokB = len(tokB['attention_mask'][0])
    nTokC = len(tokC['attention_mask'][0])
    nTokD = len(tokD['attention_mask'][0])
    nTokAll = nTokA+nTokB+nTokC+nTokD
    position_ids = torch.arange(0,nTokAll).unsqueeze(0)
    position_ids[0][nTokA+nTokB:nTokA+nTokB+nTokC] = torch.arange(nTokA,nTokA+nTokC)
    return position_ids
def get_position_ids_padded(tokA, tokB, tokC, tokD):
    nTokA = len(tokA['attention_mask'][0])
    nTokB = len(tokB['attention_mask'][0])
    nTokC = len(tokC['attention_mask'][0])
    nTokD = len(tokD['attention_mask'][0])
    nTokBCPadded = max(nTokB, nTokC)
    nTokAll = nTokA+2*nTokBCPadded+nTokD
    if nTokB<nTokBCPadded:
        # pad tokB
        tokB['input_ids'] = torch.cat((tokB['input_ids'][0], torch.tensor([tokenizer.pad_token_id]*(nTokBCPadded-nTokB)))).unsqueeze(0)
        tokB['attention_mask'] = torch.cat((tokB['attention_mask'][0], torch.zeros(nTokBCPadded-nTokB))).unsqueeze(0)
    if nTokC<nTokBCPadded:
        # pad tokC
        tokC['input_ids'] = torch.cat((tokC['input_ids'][0], torch.tensor([tokenizer.pad_token_id]*(nTokBCPadded-nTokC)))).unsqueeze(0)
        tokC['attention_mask'] = torch.cat((tokC['attention_mask'][0], torch.zeros(nTokBCPadded-nTokC))).unsqueeze(0)
    
    tokAll={'input_ids':torch.cat((tokA['input_ids'][0], tokB['input_ids'][0], tokC['input_ids'][0], tokD['input_ids'][0])).unsqueeze(0), 
            'attention_mask':torch.cat((tokA['attention_mask'][0], tokB['attention_mask'][0], tokC['attention_mask'][0], tokD['attention_mask'][0])).unsqueeze(0)}
    tokRev={'input_ids':torch.cat((tokA['input_ids'][0], tokC['input_ids'][0], tokB['input_ids'][0], tokD['input_ids'][0])).unsqueeze(0), 
            'attention_mask':torch.cat((tokA['attention_mask'][0], tokC['attention_mask'][0], tokB['attention_mask'][0], tokD['attention_mask'][0])).unsqueeze(0)}
    assert(nTokAll==len(tokAll['attention_mask'][0]))
    position_ids = torch.cat((torch.arange(0,nTokA+nTokBCPadded),torch.arange(nTokA,nTokA+nTokBCPadded),torch.arange(nTokA+nTokBCPadded,nTokA+nTokBCPadded+nTokD)))
    
    return position_ids,tokB, tokC, tokAll, tokRev
# position_ids,tokB, tokC, tokAll, tokRev=get_position_ids_padded(tokA, tokB, tokC, tokD)
# Get attention mask where tokens in tokC don't attend to tokens in tokB
def get_attention_mask_2d(tokA, tokB, tokC, tokD, tokAll=None):
    '''
    Outputs a [1,1,s,s] attention mask where s is the total number of tokens in tokA, tokB, tokC, tokD
        Values are False where tokens are masked, True where attention should be paid
    '''
    nTokA, nTokB, nTokC, nTokD = len(tokA['attention_mask'][0]), len(tokB['attention_mask'][0]), len(tokC['attention_mask'][0]), len(tokD['attention_mask'][0])
    s=nTokA+nTokB+nTokC+nTokD
    if tokAll is None:
         tokAll={'input_ids':torch.cat((tokA['input_ids'][0], tokB['input_ids'][0], tokC['input_ids'][0], tokD['input_ids'][0])).unsqueeze(0), 
            'attention_mask':torch.cat((tokA['attention_mask'][0], tokB['attention_mask'][0], tokC['attention_mask'][0], tokD['attention_mask'][0])).unsqueeze(0)}
    assert(s==len(tokAll['attention_mask'][0]))
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool))
    # Set values in causal_mask to False where seqC tokens would attend to seqB tokens
    mask = tf.Variable(causal_mask)
    
    # Create tf.Variable object filled with False of dimension (nTokC, nTokB)
    mask[nTokA+nTokB:nTokA+nTokB+nTokC, nTokA:nTokA+nTokB].assign(tf.zeros([nTokC, nTokB], tf.bool))
    
    # All tokens should ignore the padding tokens, which occur at indices where tokAll['attention_mask'][0]==0
    paddingTokenIndices = torch.where(tokAll['attention_mask'][0]==0)[0]
    for ptI in paddingTokenIndices.numpy():
        mask[:,ptI].assign(tf.zeros([s], tf.bool))

    mask=tf.convert_to_tensor([tf.Variable(mask)])
    mask=torch.tensor(mask.numpy())
    mask = mask.view(1,1,s,s)
    assert(mask.shape == (1,1,s,s))
    return mask

def testAttentionMaskEdits(seqA,seqB,seqC,seqD):
    tokA,tokB,tokC,tokD = tuple([tokenizer(input_text, return_tensors='pt').to(torch_device) for input_text in [seqA,seqB,seqC,seqD]])
    tokAll = tokenizer(seqA+seqB+seqC+seqD, return_tensors='pt').to(torch_device)
    tokRev = tokenizer(seqA+seqC+seqB+seqD, return_tensors='pt').to(torch_device)
    assert(len(tokA['attention_mask'][0]) + len(tokB['attention_mask'][0]) + len(tokC['attention_mask'][0]) + len(tokD['attention_mask'][0]) == len(tokAll['attention_mask'][0]))
    nTokA, nTokB, nTokC, nTokD = len(tokA['attention_mask'][0]), len(tokB['attention_mask'][0]), len(tokC['attention_mask'][0]), len(tokD['attention_mask'][0])
    s=nTokA+nTokB+nTokC+nTokD

    # Standard causal mask - ABCD
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool)).view(1, 1, s, s)
    generateStandard = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=causal_mask)
    outputStandard = model(**tokAll, attention_mask_2d=causal_mask)
    textStandard = tokenizer.decode(generateStandard[0], skip_special_tokens=True)

    # Edited causal mask - ABCD
    attention_mask_2d=get_attention_mask_2d(tokA, tokB, tokC, tokD)
    assert(not torch.equal(attention_mask_2d, causal_mask))
    generateEdited = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=attention_mask_2d)
    outputEdited = model(**tokAll, attention_mask_2d=attention_mask_2d)
    textEdited = tokenizer.decode(generateEdited[0], skip_special_tokens=True)

    # Standard causual mask - ACBD
    generateStandardRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=causal_mask)
    outputStandardRev = model(**tokRev, attention_mask_2d=causal_mask)
    textStandardRev = tokenizer.decode(generateStandardRev[0], skip_special_tokens=True)

    # Edited causal mask - ACBD
    attention_mask_2dRev=get_attention_mask_2d(tokA, tokC, tokB, tokD)
    assert(not torch.equal(attention_mask_2dRev, causal_mask))
    generateEditedRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=attention_mask_2dRev)
    outputEditedRev = model(**tokRev, attention_mask_2d=attention_mask_2dRev)
    textEditedRev = tokenizer.decode(generateEditedRev[0], skip_special_tokens=True)
    
    # positional edits - ABCD
    position_ids = get_position_ids(tokA, tokB, tokC, tokD)
    generateEditedPos = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=causal_mask, position_ids=position_ids)
    textEditedPos = tokenizer.decode(generateEditedPos[0], skip_special_tokens=True)
    print(f"Edited ABCD with attention and positional edits: {textEditedPos}")
    #assert(not torch.equal(outputEditedPos.logit, generateEdited.logit)) # editing position ids should change the logits
    
    # positional edits - ACBD
    position_ids_rev = get_position_ids(tokA, tokC, tokB, tokD)
    generateEditedPosRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=causal_mask, position_ids=position_ids_rev)
    textEditedPosRev = tokenizer.decode(generateEditedPosRev[0], skip_special_tokens=True)
    print(f"Edited ACBD with attention and positional edits: {textEditedPosRev}")
    #assert(not torch.equal(outputEditedPosRev.logit, generateEditedRev.logit)) # editing position ids should change the logits
    
    # Edited causal mask + positional edits - ABCD
    position_ids = get_position_ids(tokA, tokB, tokC, tokD)
    generateEditedAttentionPos = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=attention_mask_2d, position_ids=position_ids)
    textEditedAttentionPos = tokenizer.decode(generateEditedAttentionPos[0], skip_special_tokens=True)
    print(f"Edited ABCD with attention and positional edits: {textEditedAttentionPos}")
    
    # Edited causal mask + positional edits - ACBD
    position_ids_rev = get_position_ids(tokA, tokC, tokB, tokD)
    generateEditedAttentionPosRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=attention_mask_2dRev, position_ids=position_ids_rev)
    textEditedAttentionPosRev = tokenizer.decode(generateEditedAttentionPosRev[0], skip_special_tokens=True)
    print(f"Edited ACBD with attention and positional edits: {textEditedAttentionPosRev}")

    # Measure distance between different kinds of outputs
    assert(not all([torch.equal(outputEdited.attentions[i], outputStandard.attentions[i]) for i in range(len(outputEdited.attentions))])) # should not all be same, some attentions should be different!
    assert(not torch.equal(outputEdited.logits, outputStandard.logits)) # shouldn't be equal!
    cos = torch.nn.CosineSimilarity(dim=2)
    # not all 1.0 - shape (1, num_input_tokens). Note that only the logits corresponding to inputC tokens are different.
    # Do we expect the tokD logits to be different as a function of the tokC logits begin different? Or no? 
    #print(cos(outputEdited.logits, outputStandard.logits)) 
    #print(cos(outputEditedRev.logits, outputStandardRev.logits)) 
    #print(cos(outputEditedRev.logits, outputEdited.logits)) 
    #print(cos(outputStandardRev.logits, outputStandard.logits)) 

    # Note: the outputEditedRev.logits and outputEdited.logits are closer to each other than the outputStandardRev.logits and outputStandard.logits. This is what we'd hope for!
    print(f"Diff between ABCD vs ACBD output when editing attention mask: {len(tokAll.input_ids[0])-cos(outputEditedRev.logits, outputEdited.logits).sum()}")
    print(f"Diff between ABCD vs ACBD output when using standard causal attention mask: {len(tokAll.input_ids[0])-cos(outputStandardRev.logits, outputStandard.logits).sum()}")

    #torch.nn.functional.cosine_similarity(outputStandardRev.logits, outputStandard.logits)
    #torch.nn.functional.kl_div(outputStandardRev.logits, outputStandard.logits)
    s=len(seqA)+len(seqB)+len(seqC)+len(seqD)
    textStandard = textStandard[s:]
    textStandardRev = textStandardRev[s:]
    textEdited = textEdited[s:]
    textEditedRev = textEditedRev[s:]
    textEditedPos = textEditedPos[s:]
    textEditedPosRev = textEditedPosRev[s:]
    textEditedAttentionPos = textEditedAttentionPos[s:]
    textEditedAttentionPosRev = textEditedAttentionPosRev[s:]
    #print(f"Prompt: {seqA+seqB+seqC+seqD}\n\n")
    #print(f"Standard ABCD: {textStandard}\n\nEdited ABCD: {textEdited}\n\nStandard ACBD: {textStandardRev}\n\nEdited ACBD: {textEditedRev}")
    return textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev

def testAttentionPosPaddingEdits(seqA,seqB,seqC,seqD):
    # Use padding when modifying the positional encoding
    tokA,tokB,tokC,tokD = tuple([tokenizer(input_text, return_tensors='pt').to(torch_device) for input_text in [seqA,seqB,seqC,seqD]])
    tokAll = tokenizer(seqA+seqB+seqC+seqD, return_tensors='pt').to(torch_device)
    tokRev = tokenizer(seqA+seqC+seqB+seqD, return_tensors='pt').to(torch_device)
    assert(len(tokA['attention_mask'][0]) + len(tokB['attention_mask'][0]) + len(tokC['attention_mask'][0]) + len(tokD['attention_mask'][0]) == len(tokAll['attention_mask'][0]))
    nTokA, nTokB, nTokC, nTokD = len(tokA['attention_mask'][0]), len(tokB['attention_mask'][0]), len(tokC['attention_mask'][0]), len(tokD['attention_mask'][0])
    s=nTokA+nTokB+nTokC+nTokD

    # Standard causal mask - ABCD
    causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool)).view(1, 1, s, s)
    generateStandard = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=causal_mask)
    outputStandard = model(**tokAll, attention_mask_2d=causal_mask)
    textStandard = tokenizer.decode(generateStandard[0], skip_special_tokens=True)

    # Edited causal mask - ABCD
    attention_mask_2d=get_attention_mask_2d(tokA, tokB, tokC, tokD)
    assert(not torch.equal(attention_mask_2d, causal_mask))
    generateEdited = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=attention_mask_2d)
    outputEdited = model(**tokAll, attention_mask_2d=attention_mask_2d)
    textEdited = tokenizer.decode(generateEdited[0], skip_special_tokens=True)

    # Standard causual mask - ACBD
    generateStandardRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=causal_mask)
    outputStandardRev = model(**tokRev, attention_mask_2d=causal_mask)
    textStandardRev = tokenizer.decode(generateStandardRev[0], skip_special_tokens=True)

    # Edited causal mask - ACBD
    attention_mask_2dRev=get_attention_mask_2d(tokA, tokC, tokB, tokD)
    assert(not torch.equal(attention_mask_2dRev, causal_mask))
    generateEditedRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=attention_mask_2dRev)
    outputEditedRev = model(**tokRev, attention_mask_2d=attention_mask_2dRev)
    textEditedRev = tokenizer.decode(generateEditedRev[0], skip_special_tokens=True)
    
    # Reconfigure for positional edits with padding
    position_ids,tokB, tokC, tokAll, tokRev = get_position_ids_padded(tokA, tokB, tokC, tokD)
    
    # positional edits - ABCD
    generateEditedPos = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=causal_mask, position_ids=position_ids)
    textEditedPos = tokenizer.decode(generateEditedPos[0], skip_special_tokens=True)
    print(f"Edited ABCD with attention and positional edits: {textEditedPos}")
    #assert(not torch.equal(outputEditedPos.logit, generateEdited.logit)) # editing position ids should change the logits
    
    # positional edits - ACBD
    generateEditedPosRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=causal_mask, position_ids=position_ids)
    textEditedPosRev = tokenizer.decode(generateEditedPosRev[0], skip_special_tokens=True)
    print(f"Edited ACBD with attention and positional edits: {textEditedPosRev}")
    #assert(not torch.equal(outputEditedPosRev.logit, generateEditedRev.logit)) # editing position ids should change the logits
    
    # Edited causal mask + positional edits - ABCD
    attention_mask_2d=get_attention_mask_2d(tokA, tokB, tokC, tokD, tokAll)
    generateEditedAttentionPos = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=attention_mask_2d, position_ids=position_ids)
    textEditedAttentionPos = tokenizer.decode(generateEditedAttentionPos[0], skip_special_tokens=True)
    print(f"Edited ABCD with attention and positional edits: {textEditedAttentionPos}")
    
    # Edited causal mask + positional edits - ACBD
    attention_mask_2dRev=get_attention_mask_2d(tokA, tokC, tokB, tokD, tokRev)
    generateEditedAttentionPosRev = model.generate(**tokRev, max_new_tokens=10, attention_mask_2d=attention_mask_2dRev, position_ids=position_ids)
    textEditedAttentionPosRev = tokenizer.decode(generateEditedAttentionPosRev[0], skip_special_tokens=True)
    print(f"Edited ACBD with attention and positional edits: {textEditedAttentionPosRev}")

    # Measure distance between different kinds of outputs
    assert(not all([torch.equal(outputEdited.attentions[i], outputStandard.attentions[i]) for i in range(len(outputEdited.attentions))])) # should not all be same, some attentions should be different!
    assert(not torch.equal(outputEdited.logits, outputStandard.logits)) # shouldn't be equal!
    cos = torch.nn.CosineSimilarity(dim=2)
    # not all 1.0 - shape (1, num_input_tokens). Note that only the logits corresponding to inputC tokens are different.
    # Do we expect the tokD logits to be different as a function of the tokC logits begin different? Or no? 
    #print(cos(outputEdited.logits, outputStandard.logits)) 
    #print(cos(outputEditedRev.logits, outputStandardRev.logits)) 
    #print(cos(outputEditedRev.logits, outputEdited.logits)) 
    #print(cos(outputStandardRev.logits, outputStandard.logits)) 

    # Note: the outputEditedRev.logits and outputEdited.logits are closer to each other than the outputStandardRev.logits and outputStandard.logits. This is what we'd hope for!
    print(tokAll.keys())
    nTokAll = len(tokAll['input_ids'][0])
    print(f"Diff between ABCD vs ACBD output when editing attention mask: {nTokAll-cos(outputEditedRev.logits, outputEdited.logits).sum()}")
    print(f"Diff between ABCD vs ACBD output when using standard causal attention mask: {nTokAll-cos(outputStandardRev.logits, outputStandard.logits).sum()}")

    #torch.nn.functional.cosine_similarity(outputStandardRev.logits, outputStandard.logits)
    #torch.nn.functional.kl_div(outputStandardRev.logits, outputStandard.logits)
    s=len(seqA)+len(seqB)+len(seqC)+len(seqD)
    textStandard = textStandard[s:]
    textStandardRev = textStandardRev[s:]
    textEdited = textEdited[s:]
    textEditedRev = textEditedRev[s:]
    textEditedPos = textEditedPos[s:]
    textEditedPosRev = textEditedPosRev[s:]
    textEditedAttentionPos = textEditedAttentionPos[s:]
    textEditedAttentionPosRev = textEditedAttentionPosRev[s:]
    #print(f"Prompt: {seqA+seqB+seqC+seqD}\n\n")
    #print(f"Standard ABCD: {textStandard}\n\nEdited ABCD: {textEdited}\n\nStandard ACBD: {textStandardRev}\n\nEdited ACBD: {textEditedRev}")
    return textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev

seqA = 'Answer the following question after the prompt \"Answer\". Question: Consider the following pets.'
seqB = 'The dog is fluffy and brown.'
seqC = 'The cat is black and white and scraggly.'
seqD = 'Which animal is cuter? Answer:'

#testAttentionMaskEdits(seqA,seqB,seqC,seqD)

def get_mcq_prompt(stem,optionA, optionB, optionC, optionD):
    seqA = f"Answer the following multiple choice question, by first stating which of the following options fits best, and then explaining your reasoning. Here is the question: {stem}. The options are:"
    #seqB = f" A. {optionA} B. {optionB}"
    #seqC = f" C. {optionC} D. {optionD}"
    seqB = f" {optionA}, {optionB},"
    seqC = f" {optionC}, {optionD},"
    seqD = " The answer is: "
    return seqA, seqB, seqC, seqD

seqA, seqB, seqC, seqD = get_mcq_prompt("Sammy wanted to go to where the people were. Where might he go?",
                                        "race track", "populated areas", "the desert", "apartment")

#testAttentionMaskEdits(seqA,seqB,seqC,seqD)

seqA, seqB, seqC, seqD = get_mcq_prompt("To locate a choker not located in a jewelry box or boutique where would you go?",
                                        "jewelry box", "jewelry store", "boutique", "store")

#testAttentionMaskEdits(seqA,seqB,seqC,seqD)


'''
# NOTE: remove below later. Just for position id editing testing
position_ids = get_position_ids(tokA, tokB, tokC, tokD)#.view(-1,1)
generateEditedAttentionPos = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=attention_mask_2d, position_ids=position_ids)
generateEditedAttention = model.generate(**tokAll, max_new_tokens=10, attention_mask_2d=attention_mask_2d)
outputEditedAttentionPos = model(**tokAll,  attention_mask_2d=attention_mask_2d, position_ids=position_ids)
outputEditedAttention = model(**tokAll,  attention_mask_2d=attention_mask_2d)
# TODO 1/14/23 6:30pm: inputting the position ids works for the first forward attention call when position_ids with shape [65, 1] are expected, but it fails in subsequence calls.
# Fix this!!! Maybe check whether the input position_ids are the right shape, and if not, generate default position_ids?
# Within the GPT2ModelAttention.forward() function call
textEditedAttentionPos = tokenizer.decode(generateEditedAttentionPos[0], skip_special_tokens=True)
textEditedAttention = tokenizer.decode(generateEditedAttention[0], skip_special_tokens=True)
print(f"Edited ABCD with attention and positional edits: {textEditedAttentionPos}")
print(f"Edited ABCD with attention edits: {textEditedAttention}")
torch.allclose(outputEditedAttentionPos.logits, outputEditedAttention.logits)

cos = torch.nn.CosineSimilarity(dim=2)
print(cos(outputEditedAttentionPos.logits, outputEditedAttention.logits).sum())
cos = torch.nn.CosineSimilarity(dim=1)
print(cos(outputEditedAttentionPos.logits, outputEditedAttention.logits).sum)
'''