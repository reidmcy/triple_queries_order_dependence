from attention_mask_editing import get_mcq_prompt, testAttentionMaskEdits, get_position_ids_padded, testAttentionPosPaddingEdits
import json

# Read in MCQ dataset from train_rand_split.jsonl
mcq_rows = []
with open('data/train_rand_split.jsonl') as f:
    for line in f:
        mcq_rows.append(json.loads(line))
print(len(mcq_rows))

# Generate all ABCD-formatted prompts from MCQ questions
prompts = []
for row in mcq_rows:
    options=tuple([row['question']['choices'][i]['text'] for i in range(4)])
    prompt = get_mcq_prompt(row['question']['stem'],*options)
    prompts.append(prompt)
print(len(prompts))

# Generate model outputs for all prompts - approach #1 for position ids
outputs = []
for prompt in prompts[10:20]:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev = testAttentionMaskEdits(*prompt)
    outputs.append([textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt])

# Print all nicely formatted prompts and outputs
for output in outputs:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt = tuple(output)
    print(f"Prompt: {prompt}")
    print(f"Standard ABCD: {textStandard}\nStandard ACBD: {textStandardRev}")
    print(f"Attention ABCD: {textEdited}\nAttention ACBD: {textEditedRev}")
    print(f"Position ABCD: {textEditedPos}\nPosition ACBD: {textEditedPosRev}")
    print(f"Attention Position ABCD: {textEditedAttentionPos}\nAttention Position ACBD: {textEditedAttentionPosRev}\n\n")


# Generate model outputs for all prompts - approach #2 for position ids
outputs_padding = []
for prompt in prompts[10:20]:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev = testAttentionPosPaddingEdits(*prompt)
    outputs_padding.append([textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt])

for output in outputs_padding:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt = tuple(output)
    print(f"Prompt: {prompt}")
    print(f"Standard ABCD: {textStandard}\nStandard ACBD: {textStandardRev}")
    print(f"Attention ABCD: {textEdited}\nAttention ACBD: {textEditedRev}")
    print(f"Position (padded) ABCD: {textEditedPos}\nPosition (padded) ACBD: {textEditedPosRev}")
    print(f"Attention Position (padded) ABCD: {textEditedAttentionPos}\nAttention Position (padded) ACBD: {textEditedAttentionPosRev}\n\n")


# Check which prompts produce different outputs when intervening in position ids vs not intervening on position ids
for output in outputs_padding:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt = tuple(output)
    print(textStandard==textEditedPos, textStandardRev==textEditedPosRev, textEditedAttentionPos==textEditedAttentionPosRev)
    
# Analyze differences in output when using position ids with no padding (approach #1) vs position ids with padding (approach #2)
# Different for prompts 1, 5, 6, 10
for output,output_padded in zip(outputs, outputs_padding):
    _,_,_,_,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,_ = tuple(output)
    _,_,_,_,textEditedPos_padded,textEditedPosRev_padded,textEditedAttentionPos_padded,textEditedAttentionPosRev_padded,_ = tuple(output_padded)
    print(textEditedPos==textEditedPos_padded, textEditedPosRev==textEditedPosRev_padded, textEditedAttentionPos==textEditedAttentionPos_padded, textEditedAttentionPosRev==textEditedAttentionPosRev_padded)
    if not (textEditedPos==textEditedPos_padded and textEditedPosRev==textEditedPosRev_padded and textEditedAttentionPos==textEditedAttentionPos_padded and textEditedAttentionPosRev==textEditedAttentionPosRev_padded):
        print(output[-1])
        print(output[4:8])
        print(output_padded[:8])
        print()



