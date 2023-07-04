#%%
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

from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from plotting import ntensor_to_long
from jamesd_utils import projection_ratio

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1
# FIG_FILEPATH = "figs/fig1_resid_onto_H0_2_lineplot.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
# prompts = get_prompts_t(
#     n_text_prompts=N_TEXT_PROMPTS,
#     n_code_prompts=N_CODE_PROMPTS,
# ).to(device)
prompts = model.to_tokens("I went to university at Michigan").to(device)
correct_token_string = " State"
correct_token_id = model.to_tokens(correct_token_string, prepend_bos=False)[0][0].item()

#%%
# Generate names of residual stream locations
resid_names = ["blocks.0.hook_resid_pre"]
for i in range(model.cfg.n_layers):
    resid_names.append(f"blocks.{i}.hook_resid_mid")
    resid_names.append(f"blocks.{i}.hook_resid_post")

#%%
# Get the original output of H0.2
orig_logits, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: (
        "blocks.0.attn.hook_result" in name or
        name in resid_names or
        "ln_final.hook_scale" in name
    ),
    device=device,
)
orig_logits = orig_logits[:, -1, :]  # shape: (batch, d_vocab)
# del _
gc.collect()

# Select head 2 but keep the head dimension
H0_2 = einops.rearrange(
    cache["blocks.0.attn.hook_result"][:, :, 2:3, :],
    "batch pos head d_model -> head batch pos d_model",
)  # shape: (head, batch, pos, d_model)
resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

#%%
# Project the resids onto H0.2
# Mathematically:
# (a⋅b̂) * b̂ = (a⋅b / ||b||^2) * b = projection_ratio(a, b) * b
proj_resids_onto_H0_2 = projection_ratio(resids, H0_2).unsqueeze(-1) * H0_2

# Manually layernorm the resids before unembedding
mean = proj_resids_onto_H0_2.mean(dim=-1, keepdim=True)
scale = cache['ln_final.hook_scale']
proj_resids_onto_H0_2_normed = (proj_resids_onto_H0_2 - mean) / scale
"Shape: (resid, batch, pos, d_model)"

#%%
dla_logits = einops.einsum(
    proj_resids_onto_H0_2_normed[:, :,  -1, :],
    model.W_U,
    "resid batch d_model, d_model d_vocab -> resid batch d_vocab",
)

#%%
orig_prob = orig_logits.softmax(dim=-1)[..., correct_token_id].item()
new_probs = (orig_logits - dla_logits).softmax(dim=-1)[..., correct_token_id].flatten().cpu().numpy()

#%%
df = ntensor_to_long(
    projection_ratio(resids, H0_2),  # shape: (resid, batch, pos)
    value_name="projection_ratio",
    dim_names=["resid", "batch", "pos"],
)

#%%
fig, ax = plt.subplots()
ax2 = ax.twinx()

sns.lineplot(
    data=df.query("pos == pos.max()"),  # select the last token
    x="resid",
    y="projection_ratio",
    ax=ax,
)
sns.lineplot(
    x=df.resid.unique(),
    y=new_probs,
    color="C1",
    ax=ax2,
)
ax2.set_ylabel("softmax(logits-DLA)[t]")

# import psutil; psutil.virtual_memory()