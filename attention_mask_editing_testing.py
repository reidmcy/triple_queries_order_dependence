from attention_mask_editing import get_mcq_prompt, testAttentionMaskEdits, get_position_ids_padded, testAttentionPosPaddingEdits, testAttentionPosInterpolatedEdits, testAttentionPosPaddingEditsFiveOptions, get_mcq_prompt_five_options, get_mcq_prompt_two_options
import json
import pandas as pd

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
diffs = []
for prompt in prompts[10:20]:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,diff = testAttentionMaskEdits(*prompt)
    outputs.append([textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt])
    diffs.append(diff)
df = pd.DataFrame.from_records(diffs).astype(float)
print(df)
df.to_csv("mcq_kl_divergence_examples.csv")
    
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
diffs_padding = []
for prompt in prompts[10:20]:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev, diff = testAttentionPosPaddingEdits(*prompt)
    outputs_padding.append([textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt])
    diffs_padding.append(diff)
df_padding = pd.DataFrame.from_records(diffs_padding).astype(float)
print(df_padding)
df_padding.to_csv("mcq_kl_divergence_examples_padding.csv")

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



# Generate model outputs for all prompts - approach #3 for interpolated position ids
outputs_interp = []
diffs_interp = []
for prompt in prompts[10:20]:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev, diff = testAttentionPosInterpolatedEdits(*prompt)
    outputs_interp.append([textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt])
    diffs_interp.append(diff)
df_interp = pd.DataFrame.from_records(diffs_interp)
print(df_interp)
for output in outputs_interp:
    textStandard,textStandardRev,textEdited,textEditedRev,textEditedPos,textEditedPosRev,textEditedAttentionPos,textEditedAttentionPosRev,prompt = tuple(output)
    print(f"Prompt: {prompt}")
    print(f"Standard ABCD: {textStandard}\nStandard ACBD: {textStandardRev}")
    print(f"Attention ABCD: {textEdited}\nAttention ACBD: {textEditedRev}")
    print(f"Position (padded) ABCD: {textEditedPos}\nPosition (padded) ACBD: {textEditedPosRev}")
    print(f"Attention Position (padded) ABCD: {textEditedAttentionPos}\nAttention Position (padded) ACBD: {textEditedAttentionPosRev}\n\n")


# Prompt the model with 5 MCQ question options, and have the model process the 5 MCQ options in parallel.
# For a given prompt, measure all unique output text sequences (up to 10 output tokens) generated when prompting the model with any permutation of the MCQ options.
# Compare model output both with and without intervening on position ids/attention masks.
parallel_five_options={}
row_subset = mcq_rows[10:20]
for row in row_subset:
    seqA,mcq_options,seqD=get_mcq_prompt_five_options(row)
    textStandardSet,textEditedSet=testAttentionPosPaddingEditsFiveOptions(seqA,mcq_options,seqD)
    q_id = row['id']
    parallel_five_options[q_id] = {'stem':row['question']['stem'], 'standardSetResponses':list(textStandardSet), 'parallelSetResponses':list(textEditedSet), 'standardNResponses':len(textStandardSet), 'parallelNResponses':len(textEditedSet)}
# save to json file "parallel_five_options.json"
with open('parallel_five_options.json', 'w') as fp:
    json.dump(parallel_five_options, fp)
    
print([(v['standardNResponses'],v['parallelNResponses']) for k,v in parallel_five_options.items()])
print([v['standardNResponses'] for k,v in parallel_five_options.items()])
