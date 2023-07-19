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
from jamesd_utils import get_logit_diff_function

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

# Number of prompts to use for resample ablation
N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
FIG_FILEPATH = "figs/fig9_DLA_barplots.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


# %%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(prompts, dim=0).shape == prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

# %%
examples = [
    {
        "text": "It's in the shelf, either on the top or the",
        "correct": " bottom",
        "incorrect": " top",
    },
    {
        "text": "I went to university at Michigan",
        "correct": " State",
        "incorrect": " University",
    },
    {
        "text": "class MyClass:\n\tdef",
        "correct": " __",
        "incorrect": " get",
    },
]


# %%
def get_DLA_logit_diff_attn_head(example, model, layer, head):
    """
    Returns the logit difference between the correct and incorrect token, when
    apply direct logit attribution (DLA) to an attention head.
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

    _, cache = model.run_with_cache(
        token_ids,
        names_filter=lambda name: (
            name == "ln_final.hook_scale" or name == attn_layer_name
        ),
    )

    # Take only the final position
    attn_head_out = cache[attn_layer_name][0, -1, head, :]  # (dmodel)
    scale = cache["ln_final.hook_scale"][0, -1]

    # Apply final layernorm to the attn head output
    attn_head_out_normed = (
        (attn_head_out - attn_head_out.mean(dim=-1, keepdim=True)) / scale
    )

    return calc_logit_diff(attn_head_out_normed)


# %%
def get_resample_ablation_logit_diff_attn_head(example, model, layer, head):
    token_ids = model.to_tokens(example["text"])
    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])

    attn_layer_name = get_act_name("result", layer)

    # Setup resampling prompts to have the same shape as original prompt
    resampling_prompts = prompts[:, : token_ids.shape[-1]]

    # Collect corrupted activations from resampling prompts
    _, corrupted_cache = model.run_with_cache(
        resampling_prompts,
        names_filter=lambda name: name == attn_layer_name,
    )
    corrupted_activations = corrupted_cache[attn_layer_name]

    # Define hook function for patching in corrupted activations
    def resample_ablation_hook_fnc(activations, hook):
        # Patch only the last pos for the given head
        activations[:, -1, head, :] = corrupted_activations[:, -1, head, :]
        return activations

    # Repeat the original prompt to match the number of resampling prompts
    repeated_token_ids = einops.repeat(
        token_ids, "batch pos -> (repeat batch) pos", repeat=resampling_prompts.shape[0]
    )

    # Do the patching
    corrupted_logits = model.run_with_hooks(
        repeated_token_ids,
        fwd_hooks=[(attn_layer_name, resample_ablation_hook_fnc)],
    )  # (batch, pos, d_vocab)

    # Return
    return (
        corrupted_logits[:, -1, correct_token_id]
        - corrupted_logits[:, -1, incorrect_token_id]
    )


# %%
def get_logit_diff(example, model):
    token_ids = model.to_tokens(example["text"])
    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])

    logits = model(token_ids)  # (batch, pos, d_vocab)

    return logits[:, -1, correct_token_id] - logits[:, -1, incorrect_token_id]


# %%
logit_diffs_dla = []
logit_diffs_ra = []

layer, head = 0, 2
for example in examples:
    # Contribution to logit diff of H0.2, according to DLA
    logit_diffs_dla.append(
        get_DLA_logit_diff_attn_head(example, model, layer, head).item()
    )

    # Contribution to logit diff of H0.2, according to resampling ablation
    # Get the original logit diff and subtract the resampling ablation logit diff
    # to get the contribution
    orig_logit_diff = get_logit_diff(example, model)
    ra_logit_diff = get_resample_ablation_logit_diff_attn_head(
        example, model, layer, head,
    )
    logit_diff_contribution = (orig_logit_diff - ra_logit_diff).tolist()
    logit_diffs_ra.append(logit_diff_contribution)

#%%
df = pd.DataFrame()
df["contribution"] = logit_diffs_dla
df["method"] = "DLA"
df["example"] = range(len(examples))

for i, contrib in enumerate(logit_diffs_ra):
    tmp_df = pd.DataFrame()
    tmp_df["contribution"] = contrib
    tmp_df["method"] = "RA"
    tmp_df["example"] = i
    df = pd.concat([df, tmp_df])

#%%
fig, ax = plt.subplots(1, 3, figsize=(14, 6))

for i in range(len(examples)):
    sns.barplot(
        data=df.query(f"example == {i}"),
        x="method",
        y="contribution",
        errorbar=("pi", 75),
        ax=ax[i]
    )
    ax[i].axhline(0, color="black", linestyle="--")
    ax[i].set_xlabel("")
    if i == 0:
        ax[i].set_ylabel("Logit Diff Contribution")
    else:
        ax[i].set_ylabel("")
    ax[i].set_xticklabels(["Direct Logit Attribution", "Resample Ablation"])
    ax[i].set_title(
        f"Prompt: {repr(examples[i]['text'])}\n"
        f"Correct token: {repr(examples[i]['correct'])}\n"
        f"Incorrect token: {repr(examples[i]['incorrect'])}"
    )

fig.suptitle(
    f"Contribution of H0.2 to logit difference between correct and incorrect tokens,"
    f" according to DLA and resampling ablation\n"
    f"Resample ablation is done with {N_TEXT_PROMPTS + N_CODE_PROMPTS} random prompts\n"
    f"Error bars: q25 - q75"
)
fig.tight_layout()

#%%
# Save figure
fig.savefig(FIG_FILEPATH)
print(f"Saved figure to {FIG_FILEPATH}")

# %%
