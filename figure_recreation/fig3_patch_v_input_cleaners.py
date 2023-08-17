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
FIG_A_FILEPATH = "figs/fig3a_patch_v_input_resid_lineplot.jpg"
FIG_B_FILEPATH = "figs/fig3b_patch_v_input_head_barplot.jpg"

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
model.cfg.use_split_qkv_input = True
# prompts = model.to_tokens("It's in the shelf, either on the top or the").to(device)

#%%
# Generate names of residual stream locations
resid_names = ["blocks.0.hook_resid_pre"]
for i in range(model.cfg.n_layers):
    resid_names.append(f"blocks.{i}.hook_resid_mid")
    resid_names.append(f"blocks.{i}.hook_resid_post")

#%%
# Get the original output of L0H2
logits, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: (
        "blocks.0.attn.hook_result" in name or
        "blocks.2.attn.hook_result" in name or
        "blocks.2.ln1.hook_scale" in name or
        name in resid_names
    ),
    device=device,
)

# Take what we need from cache
orig_L0H2 = cache["blocks.0.attn.hook_result"][:, :, 2:3, :]  # (batch, pos, 1, d_model)
orig_L2HX = cache["blocks.2.attn.hook_result"]  # (batch, pos, head, d_model)
orig_ln_scale = cache["blocks.2.ln1.hook_scale"]  # (batch, pos, head, 1)

orig_resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

del logits, cache
gc.collect()

#%%
# Remove L0H2 from the v_input to L2HX
def hook_remove_L0H2(activations, hook):
    if hook.name == "blocks.2.hook_v_input":
        print("Did the v_input hook!")
        activations = activations - orig_L0H2
    return activations

# Use the original layernorm scale for L2HX
def hook_patch_ln_scale(scale, hook):
    if hook.name == "blocks.2.ln1.hook_scale":
        print("Did the ln_final scale hook!")
        return orig_ln_scale

# Add hooks and run with cache
model.reset_hooks()
model.add_hook("blocks.2.hook_v_input", hook_remove_L0H2, level=1)
model.add_hook("blocks.2.ln1.hook_scale", hook_patch_ln_scale, level=1)
logits, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: (
        "blocks.2.attn.hook_result" in name or
        name in resid_names
    ),
    device=device,
)
model.reset_hooks()

# Take what we need from cache
patched_H2_X = cache["blocks.2.attn.hook_result"]

patched_resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

del logits, cache
gc.collect()

#%%
df_per_head_orig = ntensor_to_long(
    projection_ratio(orig_L2HX, orig_L0H2),
    value_name="projection_ratio",
    dim_names=["batch", "pos", "head"],
)
df_per_head_patched = ntensor_to_long(
    projection_ratio(patched_H2_X, orig_L0H2),
    value_name="projection_ratio",
    dim_names=["batch", "pos", "head"],
)

#%%
orig_resids_onto_writer_prs = projection_ratio(
    orig_resids,
    einops.rearrange(orig_L0H2, "batch pos head d_model -> head batch pos d_model"),
)
patched_resids_onto_writer_prs = projection_ratio(
    patched_resids,
    einops.rearrange(orig_L0H2, "batch pos head d_model -> head batch pos d_model"),
)

df_orig_resid = ntensor_to_long(
    orig_resids_onto_writer_prs,
    value_name="projection_ratio",
    dim_names=["resid", "batch", "pos"],
)
df_orig_resid["run"] = "orig"

df_patched_resid = ntensor_to_long(
    patched_resids_onto_writer_prs,
    value_name="projection_ratio",
    dim_names=["resid", "batch", "pos"],
)
df_patched_resid["run"] = "patched"

df_resid = pd.concat([df_orig_resid, df_patched_resid], axis=0)

#%%
resid_names_plot = ["resid_pre0"]
for i in range(model.cfg.n_layers):
    resid_names_plot.append(f"resid_mid{i}")
    resid_names_plot.append(f"resid_post{i}")
#%%
fig_a, ax_a = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data=df_resid,
    x="resid",
    y="projection_ratio",
    estimator="median",
    hue="run",
    errorbar=("pi", 75),
    ax=ax_a,
)
ax_a.set_title(
    f"Projection of residual stream onto L0H2 without/with patching",
    fontsize=14,
)
ax_a.set_ylabel("PR(resid, L0H2)", fontsize=14)
ax_a.set_xlabel("")
ax_a.set_xticks(
    ticks=range(len(resid_names_plot)),
    labels=resid_names_plot,
)
ax_a.legend(
    title="Run",
    labels=["Original", "Patched"],
    handles=[
        mlines.Line2D([0], [0], color="C0", lw=2),
        mlines.Line2D([0], [0], color="C1", lw=2),
    ]
);

#%%
# Save figure
fig_a.savefig(FIG_A_FILEPATH, bbox_inches="tight")
print(f"Saved figure to {FIG_A_FILEPATH}")

#%%
fig_b, ax_b = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Left subplot
sns.barplot(
    data=df_per_head_orig,
    x="head",
    y="projection_ratio",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax_b[0],
)
ax_b[0].set_title(
    f"Projections of L2HX onto L0H2 without patching",
    fontsize=16,
)
ax_b[0].set_ylabel("PR(L2HX, L0H2)", fontsize=14)
ax_b[0].set_xlabel("")
ax_b[0].set_xticks(
    ticks=range(model.cfg.n_heads),
    labels=[f"L2H{h}" for h in range(model.cfg.n_heads)],
    fontsize=14,
)

# Right subplot
sns.barplot(
    data=df_per_head_patched,
    x="head",
    y="projection_ratio",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax_b[1],
)
ax_b[1].set_title(
    f"Projections of L2HX onto L0H2 with patching",
    fontsize=16,
)
ax_b[1].set_ylabel("")
ax_b[1].set_xlabel("")
ax_b[1].set_xticks(
    ticks=range(model.cfg.n_heads),
    labels=[f"L2H{h}" for h in range(model.cfg.n_heads)],
    fontsize=14,
)

fig_b.tight_layout()

#%%
fig_b.savefig(FIG_B_FILEPATH, bbox_inches="tight")
print(f"Saved figure to {FIG_B_FILEPATH}")


"""
# NOTE

# We think that hook_v_input has not been layer normed yet

# This shows that the norms are different
einops.rearrange(
    cache['blocks.2.hook_v_input'],
    "batch pos head d_model -> head batch pos d_model",
).norm(dim=-1)

# After passing through a layer norm, all the norms are the same
einops.rearrange(
    model.blocks[2].ln1(cache['blocks.2.hook_v_input']),
    "batch pos head d_model -> head batch pos d_model",
).norm(dim=-1)
"""
