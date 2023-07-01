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

from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from plotting import ntensor_to_long
from jamesd_utils import projection_ratio, projection_value

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1
FIG_FILEPATH = "figs/fig8_bos_nonbos_proj_ratios.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


#%%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(prompts, dim=0).shape == prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
# Define hook points to cache
hook_names = [
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_z",
    "blocks.0.attn.hook_result",
    "blocks.2.attn.hook_result",
]

# Run a forward pass and cache selected activations
_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: name in hook_names,
    device=device,
)

# Delete logits and garbage collect
del _
gc.collect()
torch.cuda.empty_cache()

#%%
# Double check calculation of `z` from `pattern` and `v` for H0.2
pattern_from_cache = cache["blocks.0.attn.hook_pattern"][:, 2, :, :]
v_from_cache = cache["blocks.0.attn.hook_v"][:, :, 2, :]
z_from_cache = cache["blocks.0.attn.hook_z"][:, :, 2, :]

z = einops.einsum(
    pattern_from_cache,  # (batch posQ posK)
    v_from_cache,  # (batch pos d_head)
    "batch posQ posK, batch posK d_head -> batch posQ d_head",
)
assert torch.allclose(z, z_from_cache, atol=1e-5)

# Double check calculation of `result` from `z` and `W_O` for H0.2
W_O_H0_2 = model.W_O[0, 2]  # (d_head, d_model)
result_from_cache = cache["blocks.0.attn.hook_result"][:, :, 2, :]  # (batch, pos, d_model)

result = einops.einsum(
    z_from_cache,
    W_O_H0_2,
    "batch pos d_head, d_head d_model -> batch pos d_model",
)
assert torch.allclose(result, result_from_cache, atol=1e-5)

#%%
# Slice BOS and non-BOS vectors for H0.2
pattern_bos = cache["blocks.0.attn.hook_pattern"][:, 2, :, 0:1]  # slice BOS but keep dim
pattern_nonbos = cache["blocks.0.attn.hook_pattern"][:, 2, :, 1:]
v_bos = cache["blocks.0.attn.hook_v"][:, 0:1, 2, :]
v_nonbos = cache["blocks.0.attn.hook_v"][:, 1:, 2, :]

# Calculate decomposed zs
z_bos = einops.einsum(
    pattern_bos,  # (batch posQ posK)
    v_bos,  # (batch pos d_head)
    "batch posQ posK, batch posK d_head -> batch posQ d_head",
)
z_nonbos = einops.einsum(
    pattern_nonbos,  # (batch posQ posK)
    v_nonbos,  # (batch pos d_head)
    "batch posQ posK, batch posK d_head -> batch posQ d_head",
)

# Calculate decomposed results from decomposed zs
result_bos = einops.einsum(
    z_bos,
    W_O_H0_2,
    "batch pos d_head, d_head d_model -> batch pos d_model",
)
result_nonbos = einops.einsum(
    z_nonbos,
    W_O_H0_2,
    "batch pos d_head, d_head d_model -> batch pos d_model",
)

# Validate decompositions
assert torch.allclose(z_bos + z_nonbos, z_from_cache, atol=1e-5)
assert torch.allclose(result_bos + result_nonbos, result_from_cache, atol=1e-5)

#%%
# Get and rearrange vectors to project from and onto
# All shapes: (head, batch, pos, d_model)
result_H0_2_bos = einops.rearrange(result_bos, "batch pos d_model -> 1 batch pos d_model")
result_H0_2_nonbos = einops.rearrange(result_nonbos, "batch pos d_model -> 1 batch pos d_model")
results_H2_X = einops.rearrange(
    cache["blocks.2.attn.hook_result"], "batch pos head d_model -> head batch pos d_model"
)

# Create BOS dataframe
df_bos = ntensor_to_long(
    projection_ratio(results_H2_X, result_H0_2_bos),
    value_name="projection_ratio",
    dim_names=["L2_head", "batch", "pos"]
)
df_bos["projection_value"] = projection_value(results_H2_X, result_H0_2_bos).flatten().cpu().numpy()

# Create non-BOS dataframe
df_nonbos = ntensor_to_long(
    projection_ratio(results_H2_X, result_H0_2_nonbos),
    value_name="projection_ratio",
    dim_names=["L2_head", "batch", "pos"]
)
df_nonbos["projection_value"] = projection_value(results_H2_X, result_H0_2_nonbos).flatten().cpu().numpy()

#%%
sns.set(font_scale=1.2)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: BOS
sns.lineplot(
    data=df_bos.query("pos != 1023").groupby(["batch", "pos"]).sum().reset_index(),
    x="pos",
    y="projection_ratio",
    errorbar=("pi", 75),
    ax=ax[0],
)
ax[0].axhline(0, color="black", linestyle="--")
ax[0].set_ylabel("Projection Ratio")
ax[0].set_title("Projection of sum(H2.X) onto H0.2 BOS only (error bars: q25 - q75)")

# Subplot 2: non-BOS
sns.lineplot(
    data=df_nonbos.query("pos != 1023").groupby(["batch", "pos"]).sum().reset_index(),
    x="pos",
    y="projection_ratio",
    errorbar=("pi", 75),
    ax=ax[1],
)
ax[1].axhline(0, color="black", linestyle="--")
ax[1].set_xlabel("Position")
ax[1].set_ylabel("Projection Ratio")
ax[1].set_title("Projection of sum(H2.X) onto H0.2 non-BOS only (error bars: q25 - q75)")

fig.tight_layout()

#%%
# Save figure
fig.savefig(FIG_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_FILEPATH)
