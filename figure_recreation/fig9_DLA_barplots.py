#%%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path
parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import gc
from functools import partial

import torch
import numpy as np
import pandas as pd
import einops

from typing import Tuple
from jaxtyping import Float
from torch import Tensor

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from load_data import get_prompts_t

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 80
N_CODE_PROMPTS = 20
FIG_FILEPATH = "figs/fig9_DLA_barplots.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


#%%
prompts = get_prompts_t().to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(prompts, dim=0).shape == prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
def resample_ablation(activation, hook, corrupted_activation, head):
    """activation: (batch, pos, head, dmodel)"""
    activation[:, -1, head, :] = corrupted_activation[hook.name][:, -1, head, :]
    return activation

#%%
examples = [
    {'text': "It's in the shelf, either on the top or the", 'answer': ' bottom'},
    {'text': "I went to university at Michigan", 'answer': ' State'},
    {'text': "class MyClass:\n\tdef", 'answer': ' __'},
]

#%%
prob_diffs = []
for example in examples:
    text = example['text']
    answer = example['answer']

    ablated_results = []
    ori_tokens = model.to_tokens(text)
    answer_token_id = model.to_tokens(answer, prepend_bos=False)[0][0].item()

    n_prompts = N_TEXT_PROMPTS + N_CODE_PROMPTS
    for random_prompt_idx in range(n_prompts):
        corrupted_tokens = prompts[random_prompt_idx:random_prompt_idx+1, :ori_tokens.shape[-1]]

        # show original probability
        original_logits = model(ori_tokens, return_type="logits")
        ori_prob, ori_token = torch.max(torch.softmax(original_logits[0, -1, :], dim=-1), dim=-1)
        ori_prob, ori_token = ori_prob.item(), ori_token.item()

        _, corrupted_activation = model.run_with_cache(corrupted_tokens)

        layer, head = 0, 2
        hook_fnc = partial(resample_ablation, corrupted_activation=corrupted_activation, head=head)
        ablated_logits = model.run_with_hooks(
            ori_tokens,
            return_type="logits",
            fwd_hooks=[(get_act_name('result', layer), hook_fnc)]
        )

        # get the token with highest probability, get the probability as well
        ablated_prob = ablated_logits.softmax(dim=-1)[0, -1, answer_token_id].item()
        ablated_results.append(ori_prob - ablated_prob)

    prob_diffs.append(ablated_results)

#%%
del original_logits, ablated_logits, corrupted_activation
gc.collect()

#%%
df_ablated = pd.DataFrame()
for i, _ in enumerate(examples):
    tmp_df = pd.DataFrame()
    tmp_df['prob_diff'] = prob_diffs[i]
    tmp_df['example'] = i
    df_ablated = pd.concat([df_ablated, tmp_df])
df_ablated["type"] = "ablation"

#%%
def get_H02_prob_contribution_by_DLA(
    prompt: str, answer: str, model: HookedTransformer
)-> Tuple[float, float]:

    layer, head = 0, 2
    logits, cache = model.run_with_cache(prompt)

    # select last position
    logits: Float[Tensor, "batch dvocab"] = logits[:, -1, :]

    # calculate direct logits attribution
    attn_out_H02: Float[Tensor, "batch dmodel"] = cache["result", layer][:, -1, head, :]

    sf = cache["ln_final.hook_scale"][0, -1]
    direct_logits: Float[Tensor, "batch dvocab"] = einops.einsum(
        (attn_out_H02 - attn_out_H02.mean(dim=-1)) / sf,
        model.W_U,
        "batch dmodel, dmodel dvocab -> batch dvocab",
    )

    token_idx = model.tokenizer(answer)["input_ids"]
    prob_new = (logits - direct_logits).softmax(dim=-1)[0, token_idx].item()
    prob_ori = logits.softmax(dim=-1)[0, token_idx].item()
    prob_diff = prob_ori - prob_new

    print(f"Prompt: {prompt}")
    print(
        f"Original prob: {prob_ori:.2f}, Prob if remove DLA: {prob_new:.2f}, Prob diff: {prob_diff:.2f}"
    )
    print()

    return prob_ori, prob_diff

#%%
prob_diffs_DLA = []
orig_probs = []
for example in examples:
    text = example['text']
    answer = example['answer']

    prob_ori, prob_diff = get_H02_prob_contribution_by_DLA(text, answer, model)
    prob_diffs_DLA.append(prob_diff)
    orig_probs.append(prob_ori)

#%%
df_dla = pd.DataFrame()
df_dla['prob_diff'] = prob_diffs_DLA
df_dla['example'] = np.arange(len(examples))
df_dla["type"] = "DLA"

df = pd.concat([df_ablated, df_dla])

#%%
fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

for i, example in enumerate(examples):
    sns.barplot(
        data=df[df['example'] == i],
        x="type",
        y="prob_diff",
        estimator="median",
        errorbar=("pi", 75),
        ax=ax[i],
    )
    ax[i].set_title(
        f"Prompt {i+1}\n"
        f"Original Probabilty: {orig_probs[i]:.2f}"
    )
    ax[i].set_xlabel("")
    if i == 0:
        ax[i].set_ylabel("Probability Contribution")
    else:
        ax[i].set_ylabel("")
    ax[i].set_xticklabels(["Resample Ablation", "DLA"])
fig.tight_layout()

#%%
# Save figure
fig.savefig(FIG_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_FILEPATH)
