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
from jamesd_utils import projection_ratio, projection_ratio_cartesian

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1
FIG_FILEPATH = "figs/fig1_resid_onto_head_lineplots.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


#%%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)
# prompts = model.to_tokens("It's in the shelf, either on the top or the").to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(prompts, dim=0).shape == prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
# Generate names of residual stream locations
resid_names = ["blocks.0.hook_resid_pre"]
for i in range(model.cfg.n_layers):
    resid_names.append(f"blocks.{i}.hook_resid_mid")
    resid_names.append(f"blocks.{i}.hook_resid_post")

#%%
# Get the original output of H0.2
_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: (
        "blocks.0.attn.hook_result" in name or
        name in resid_names
    ),
    device=device,
)
# Select head 2 but keep the head dimension
L0HX = einops.rearrange(
    cache["blocks.0.attn.hook_result"][:, :, :, :],
    "batch pos head d_model -> head batch pos d_model",
)  # shape: (head, batch, pos, d_model)
resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

del _
gc.collect()

#%%
df = ntensor_to_long(
    projection_ratio_cartesian(resids, L0HX),
    value_name="projection_ratio",
    dim_names=["resid", "head", "batch", "pos"],
)
df["head"] = df["head"].astype("category")

resid_names_plot = ["resid_pre0"]
for i in range(model.cfg.n_layers):
    resid_names_plot.append(f"resid_mid{i}")
    resid_names_plot.append(f"resid_post{i}")

#%%
fig, ax = plt.subplots(2, 1, figsize=(11, 10), sharex=True, sharey=True)

# Top subplot
sns.lineplot(
    data=df,
    x="resid",
    y="projection_ratio",
    hue="head",
    estimator="median",
    errorbar=None,
    ax=ax[0],
)
ax[0].set_title(
    "Projections of residual stream onto L0HX",
    fontsize=16,
)
ax[0].set_ylabel("PR(resid, L0HX)", fontsize=14)

# Bottom subplot
sns.lineplot(
    data=df.query("head == 2"),
    x="resid",
    y="projection_ratio",
    estimator="median",
    errorbar=("pi", 75),
    color="C2",
    ax=ax[1],
)
ax[1].set_title(
    "Projections of residual stream onto L0H2",
    fontsize=16,
)
ax[1].set_ylabel("PR(resid, L0H2)", fontsize=14)
ax[1].set_xlabel("")
ax[1].set_xticks(
    ticks=range(len(resid_names_plot)),
    labels=resid_names_plot,
    fontsize=12,
)

fig.tight_layout();

#%%
fig.savefig(FIG_FILEPATH, bbox_inches="tight")
print(f"Saved figure to {FIG_FILEPATH}")
