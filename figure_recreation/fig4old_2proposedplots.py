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
from transformer_lens.utils import get_act_name, test_prompt
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

FIG_FILEPATH = "figs/fig4_modified_DLA_barplots.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


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
    }
]

# %%
resid_names = ["blocks.0.hook_resid_pre"]
for i in range(model.cfg.n_layers):
    resid_names.append(f"blocks.{i}.hook_resid_mid")
    resid_names.append(f"blocks.{i}.hook_resid_post")

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
            name in resid_names
            or name == "blocks.0.attn.hook_result"
            or name == "blocks.2.attn.hook_result"
            or name == "ln_final.hook_scale"
            or name == "ln_final.hook_normalized"
        ),
    )

    # Get activations from the cache
    resids = torch.stack(
        [cache[name] for name in resid_names], dim=0
    )  # (resid, batch, pos, d_model)
    L0H2 = cache["blocks.0.attn.hook_result"][:, :, 2, :]  # (batch, pos, d_model)
    L2HX = einops.reduce(
        cache["blocks.2.attn.hook_result"],
        "batch pos head d_model -> batch pos d_model",
        "sum",
    )
    scale = cache["ln_final.hook_scale"]

    apply_ln = lambda x: (x - x.mean(dim=-1, keepdim=True)) / scale
    resids_ln = apply_ln(resids)
    L0H2_ln = apply_ln(L0H2)
    L2HX_ln = apply_ln(L2HX)
    
    # Check that manual layernorming is correct
    assert torch.allclose(cache["ln_final.hook_normalized"], resids_ln[-1], atol=1e-5)

    return (
        resids_ln[:, 0, -1, :] @ logit_diff_direction,
        projection_ratio(resids[:, 0, -1, :], L0H2[0, -1, :].unsqueeze(0)),
        L0H2_ln[0, -1, :] @ logit_diff_direction,
        L2HX_ln[0, -1, :] @ logit_diff_direction,
    )

# %%
logit_lens = []
proj_ratios = []
L0H2_dlas = []
L2HX_dlas = []

for example in examples:
    ll, pr, L0H2_dla, L2HX_dla = get_results(model, example)
    logit_lens.append(ll)
    proj_ratios.append(pr)
    L0H2_dlas.append(L0H2_dla)
    L2HX_dlas.append(L2HX_dla)

# %%
df_ll = pd.DataFrame()
for i, ll in enumerate(logit_lens):
    df_ll[f"Prompt {i+1}"] = ll.numpy()

df_pr = pd.DataFrame()
for i, pr in enumerate(proj_ratios):
    df_pr[f"Prompt {i+1}"] = pr.numpy()

df_ll.plot(title="Logits Lens", figsize=(8, 4))
df_pr.plot(title="PR(resid, L0H2)", figsize=(8, 4))

# %%
for i, (d1, d2) in enumerate(zip(L0H2_dlas, L2HX_dlas)):
    print(f"Prompt {i+1} L0H2/L2HX DLAs: {d1:.3f}, {d2:.3f}")




# %%
# ex = examples[3]
# test_prompt(ex["text"], ex["correct"], model, prepend_space_to_answer=False)