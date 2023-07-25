# %%
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
from jamesd_utils import get_logit_diff_function, projection_ratio

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

IPSUM = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
# IPSUM = ""

FIG_FILEPATH = "figs/fig9new_DLA_barplots.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

# %%
examples = [
    {
        # "text": "It's in the shelf, either on the top or the",
        "text": IPSUM + " It's in the shelf, either on the top or the",
        "correct": " bottom",
        "incorrect": " top",
    },
    {
        # "text": "I went to university at Michigan",
        "text": 2 * IPSUM + " I went to university at Michigan",
        "correct": " State",
        "incorrect": " University",
    },
    {
        # "text": "class MyClass:\n\tdef",
        "text": IPSUM + " class MyClass:\n\tdef",
        "correct": " __",
        # "incorrect": " get",
        "incorrect": " on",
    },
]

# %%
def get_DLA_logit_diff_attn_head(example, model, layer, head, modified=False):
    """
    Returns the logit difference between the correct and incorrect token, when
    apply direct logit attribution (DLA) to an attention head.

    when modified==True, project the final resid post onto the attn out before DLA
    """
    token_ids = model.to_tokens(example["text"])
    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])

    calc_logit_diff = get_logit_diff_function(
        model,
        correct_token_id,
        incorrect_token_id,
    )

    attn_layer_name = get_act_name("result", layer)
    final_resid_post_name = get_act_name("resid_post", model.cfg.n_layers - 1)

    _, cache = model.run_with_cache(
        token_ids,
        names_filter=lambda name: any([
            name == "ln_final.hook_scale",
            name == attn_layer_name,
            name == final_resid_post_name, 
        ]),
    )

    # Take only the final position
    attn_head_out = cache[attn_layer_name][0, -1, head, :]  # (dmodel)
    final_resid_post = cache[final_resid_post_name][0, -1, :]  # (dmodel)
    scale = cache["ln_final.hook_scale"][0, -1]

    # Project the final resid post onto the attn out if modified==True
    if modified:
        proj_ratio = projection_ratio(final_resid_post, attn_head_out)
        print(proj_ratio)
        attn_head_out = proj_ratio * attn_head_out


    # Apply final layernorm to the attn head output
    attn_head_out_normed = (
        (attn_head_out - attn_head_out.mean(dim=-1, keepdim=True)) / scale 
    )

    return calc_logit_diff(attn_head_out_normed)

# %%
logit_diffs = []
logit_diffs_dla = []
logit_diffs_dla_modified = []

layer, head = 0, 2
for example in examples:
    # Get logit diffs
    token_ids = model.to_tokens(example["text"])
    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])
    logits = model(token_ids)[0, -1]
    logit_diff = (logits[correct_token_id] - logits[incorrect_token_id]).item()
    logit_diffs.append(logit_diff)

    # Contribution to logit diff of H0.2, according to DLA
    logit_diffs_dla.append(
        get_DLA_logit_diff_attn_head(example, model, layer, head, modified=False).item()
    )
    logit_diffs_dla_modified.append(
        get_DLA_logit_diff_attn_head(example, model, layer, head, modified=True).item()
    )

# %%
df = pd.DataFrame()
df["contribution"] = logit_diffs_dla
df["method"] = "DLA"
df["example"] = range(len(examples))

df2 = pd.DataFrame()
df2["contribution"] = logit_diffs_dla_modified
df2["method"] = "DLA modified"
df2["example"] = range(len(examples))
df = pd.concat([df, df2])

#%%
fig, ax = plt.subplots(1, 3, figsize=(14, 6))

for i in range(len(examples)):
    sns.barplot(
        data=df.query(f"example == {i}"),
        x="method",
        y="contribution",
        ax=ax[i]
    )
    ax[i].axhline(0, color="black", linestyle="--")
    ax[i].set_xlabel("")
    if i == 0:
        ax[i].set_ylabel("Logit Diff Contribution")
    else:
        ax[i].set_ylabel("")
    ax[i].set_xticklabels(["DLA", "Modified DLA"], fontsize=12)
    ax[i].set_title(
        # f"Prompt: {repr(examples[i]['text'])}\n"
        f"Prompt: {repr(examples[i]['text'].replace(IPSUM, ''))}\n"
        f"Correct token: {repr(examples[i]['correct'])}\n"
        f"Incorrect token: {repr(examples[i]['incorrect'])}\n"
        f"Original logit difference: {logit_diffs[i]:.2f}"
    )

fig.suptitle(
    f"Contribution of H0.2 to logit difference according to DLA and modified DLA",
    fontsize=16,
)
fig.tight_layout()

#%%
# Save figure
fig.savefig(FIG_FILEPATH)
print(f"Saved figure to {FIG_FILEPATH}")
