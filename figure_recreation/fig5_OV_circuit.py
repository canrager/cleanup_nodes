# %%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path

parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import gc
import torch
import einops
import pandas as pd
import numpy as np

from tqdm.auto import trange
from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from jamesd_utils import get_logit_diff_function

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

IPSUM = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
# IPSUM = ""

N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
FIG_A_FILEPATH = "figs/fig4a_DLA_resample_ablation.jpg"
FIG_B_FILEPATH = "figs/fig4b_DLA_resample_ablation.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


# %%
rand_prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(rand_prompts, dim=0).shape == rand_prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

# %%
examples = [
    {
        "text": 4 * IPSUM + " It's in the cupboard, either on the top or the",
        "correct": " bottom",
        "incorrect": " top",
    },
    {
        "text": 5 * IPSUM + " I went to university at Michigan",
        "correct": " State",
        "incorrect": " University",
    },
    {
        "text": IPSUM + " class MyClass:\n\tdef",
        "correct": " __",
        "incorrect": " on",
    },
    {
        "text": 6 * IPSUM + "The church I go to is the Seventh-day Adventist",
        "correct": " Church",
        "incorrect": " Advent",
    },
]


# %%
def get_results(model, example):
    token_ids = model.to_tokens(example["text"])
    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])
    logit_diff_direction = (
        model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id]
    )  # (d_model,)

    _, cache = model.run_with_cache(
        token_ids,
        names_filter=lambda name: (
            name == "blocks.0.attn.hook_result"
            or name == "blocks.2.ln1.hook_scale"
            or name == "ln_final.hook_scale"
        ),
    )

    # Get relevant activations from cache
    L0H2 = cache["blocks.0.attn.hook_result"][:, :, 2, :]  # (batch, pos, d_model)
    scale_layer2 = cache["blocks.2.ln1.hook_scale"]
    scale_final = cache["ln_final.hook_scale"]

    # Create layernorm functions and apply
    apply_ln_layer2 = lambda x: (x - x.mean(dim=-1, keepdim=True)) / scale_layer2
    apply_ln_final = lambda x: (x - x.mean(dim=-1, keepdim=True)) / scale_final


    # Get V-composition of L0H2 to L2HX
    L2HX_W_OV = einops.einsum(
        model.W_V[2, 2:],  # Layer 2, use heads 2 to 7 inclusive
        model.W_O[2, 2:],
        "head d_model_V d_head, head d_head d_model_O -> head d_model_V d_model_O",
    )

    L0H2_ln_layer2 = apply_ln_layer2(L0H2)

    v_comp = einops.einsum(
        L0H2_ln_layer2,
        L2HX_W_OV,
        "batch pos d_model_V, head d_model_V d_model_O -> batch pos d_model_O",
    )

    # Apply final layernorms
    L0H2_ln_final = apply_ln_final(L0H2)
    v_comp_ln_final = apply_ln_final(v_comp)

    return (
        L0H2_ln_final[0, -1, :] @ logit_diff_direction,
        v_comp_ln_final[0, -1, :] @ logit_diff_direction,
    )

# %%
dlas_L0H2 = []
dlas_v_comp = []
for i in trange(len(examples)):
    dla1, dla2 = get_results(model, examples[i])
    dlas_L0H2.append(dla1)
    dlas_v_comp.append(dla2)

# %%
fig, ax = plt.subplots(2, 2, figsize=(10,6))

for i in range(len(examples)):
    r, c = divmod(i, 2)

    # Plot data
    sns.barplot(
        x=["L0H2", "V-Comp L0H2 -> L2HX"],
        y=[dlas_L0H2[i], dlas_v_comp[i]],
        ax=ax[r, c],
    )

    # Plot aesthetics
    ax[r, c].set_title(examples[i]["correct"])
    if c == 0:
        ax[r, c].set_ylabel("Direct Logit Attribution")

plt.tight_layout()
