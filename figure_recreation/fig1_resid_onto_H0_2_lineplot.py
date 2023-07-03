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
import matplotlib.lines as mlines
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
FIG_FILEPATH = "figs/fig1_resid_onto_H0_2_lineplot.jpg"

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
H0_2 = einops.rearrange(
    cache["blocks.0.attn.hook_result"][:, :, 2:3, :],
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
    projection_ratio(resids, H0_2),
    value_name="projection_ratio",
    dim_names=["resid", "batch", "pos"],
)

resid_names_plot = ["resid_pre0"]
for i in range(model.cfg.n_layers):
    resid_names_plot.append(f"resid_mid{i}")
    resid_names_plot.append(f"resid_post{i}")

#%%
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data=df,
    x="resid",
    y="projection_ratio",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax,
)
ax.set_title(
    f"Projection of Residual Stream onto H0.2\n"
    f"Median across batch (n={prompts.shape[0]}) and position (n={prompts.shape[1]})\n"
    f"Error bars: q25 - q75"
)
ax.set_ylabel("Projection Ratio")
ax.set_xlabel("")
ax.set_xticks(
    ticks=range(len(resid_names_plot)),
    labels=resid_names_plot,
    rotation=-20,
);

#%%
fig.savefig(FIG_FILEPATH, bbox_inches="tight")
print(f"Saved figure to {FIG_FILEPATH}")
