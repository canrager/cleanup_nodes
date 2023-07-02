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

from plotly.graph_objs.layout._shape import Shape
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 1
N_CODE_PROMPTS = 0
FIG_FILEPATH = "figs/fig3_patch_v_input_cleaners.jpg"

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
# Get the original output of H0.2
orig_logits, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: (
        "blocks.0.attn.hook_result" in name or
        name in resid_names
    ),
    device=device,
)
# Select head 2 but keep the head dimension
orig_H0_2 = cache["blocks.0.attn.hook_result"][:, :, 2:3, :]  # (batch, pos, 1, d_model)

orig_loss = torch.nn.functional.cross_entropy(
    einops.rearrange(orig_logits[:, :-1, :], "batch pos d_vocab -> (batch pos) d_vocab"),
    einops.rearrange(prompts[:, 1:], "batch pos -> (batch pos)"),
    reduction="none",
)

orig_resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

del orig_logits, cache
gc.collect()

#%%
def hook_remove_H0_2(activations, hook):
    if hook.name == "blocks.2.hook_v_input":
        print("Did the hook!")
        activations = activations - orig_H0_2
    return activations

model.reset_hooks()
model.add_hook("blocks.2.hook_v_input", hook_remove_H0_2, level=1)
patched_logits, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: (
        "blocks.2.attn.hook_result" in name or
        name in resid_names
    ),
    device=device,
)

patched_loss = torch.nn.functional.cross_entropy(
    einops.rearrange(patched_logits[:, :-1, :], "batch pos d_vocab -> (batch pos) d_vocab"),
    einops.rearrange(prompts[:, 1:], "batch pos -> (batch pos)"),
    reduction="none",
)

patched_H2_X = cache["blocks.2.attn.hook_result"]
patched_resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

del patched_logits, cache
gc.collect()

#%%
df_head = ntensor_to_long(
    projection_ratio(patched_H2_X, orig_H0_2),
    value_name="projection_ratio",
    dim_names=["batch", "pos", "head"],
)

#%%
sns.barplot(
    data=df_head,
    x="head",
    y="projection_ratio",
    errorbar=("pi", 75),
)

#%%
orig_resids_onto_writer_prs = projection_ratio(
    orig_resids,
    einops.rearrange(orig_H0_2, "batch pos head d_model -> head batch pos d_model"),
)
patched_resids_onto_writer_prs = projection_ratio(
    patched_resids,
    einops.rearrange(orig_H0_2, "batch pos head d_model -> head batch pos d_model"),
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
sns.lineplot(
    data=df_resid,
    x="resid",
    y="projection_ratio",
    hue="run",
    errorbar=("pi", 75),
)

#%%

#import psutil; psutil.virtual_memory()

"""
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