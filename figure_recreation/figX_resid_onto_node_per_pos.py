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

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1
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
        "blocks.2.attn.hook_result" in name or
        name in resid_names
    ),
    device=device,
)
# Select head 2 but keep the head dimension
H0_2 = einops.rearrange(
    cache["blocks.0.attn.hook_result"][:, :, 2:3, :],
    "batch pos head d_model -> head batch pos d_model",
)  # shape: (head, batch, pos, d_model)

H2_X = einops.rearrange(
    cache["blocks.2.attn.hook_result"],
    "batch pos head d_model -> head batch pos d_model",
)  # shape: (head, batch, pos, d_model)

# Stack resids
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
df_wide = df.pivot(index="resid", columns=["batch", "pos"], values="projection_ratio")

#%%
# Plot a resid line for each pos, for a given batch
df_wide[0].iloc[:, 1:].plot(legend=False, figsize=(10, 4), title="Batch 0 (code), pos per line (pos 0 omitted)")
df_wide[1].iloc[:, 1:].plot(legend=False, figsize=(10, 4), title="Batch 1 (c4), pos per line (pos 0 omitted)")
df_wide[2].iloc[:, 1:].plot(legend=False, figsize=(10, 4), title="Batch 2 (c4), pos per line (pos 0 omitted)")

#%%
# Bin pos, and plot a resid line for each pos bin, for a given batch
n_lines = 24
df["pos_group"] = pd.cut(df.pos, bins=n_lines)

for i in range(0, n_lines, 4):
    filtered_df = (
        df.query("batch == 1")
        [df.pos_group.isin(df.pos_group.unique()[i:i+5])]
    )
    filtered_df["pos_group"] = filtered_df["pos_group"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=filtered_df,
        x="resid",
        y="projection_ratio",
        hue="pos_group",
        errorbar=("pi", 75)
    )
    ax.set_ylim([-0.8, 1.3])
