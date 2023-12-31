#%%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path
parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import torch
import einops

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

N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
WRITER_NAME = "H0.2"
FIG_FILEPATH_A = "figs/fig4a_H2_X_onto_H0_2_lineplots.jpg"
FIG_FILEPATH_B = "figs/fig4b_H2_X_onto_H0_2_lineplots.jpg"

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
# Run a forward pass and cache selected activations
hook_names = ["blocks.0.attn.hook_result", "blocks.2.attn.hook_result"]

_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: name in hook_names,
    device=device,
)

#%%
writer_cache = cache["blocks.0.attn.hook_result"][:, :, 2, :].unsqueeze(0)  # (head, batch, pos, d_model)
cleanup_cache = einops.rearrange(
    cache["blocks.2.attn.hook_result"],
    "batch pos head d_model -> head batch pos d_model",
) # (head, batch, pos, d_model)

df = ntensor_to_long(
    projection_ratio(cleanup_cache, writer_cache),
    value_name="projection_ratio",
    dim_names=["head", "batch", "pos"],
)
df["projection_value"] = projection_value(cleanup_cache, writer_cache).cpu().numpy().flatten()
df["head"] = df["head"].astype("category")

df_sum = df.groupby(["batch", "pos"]).sum(numeric_only=True).reset_index()

#%%
fig, ax = plt.subplots(figsize=(12, 7))

sns.lineplot(
    data=df.groupby(["head", "pos"]).median().reset_index(),
    x="pos",
    y="projection_ratio",
    estimator="median",
    hue="head",
    ax=ax,
)
ax.axhline(0, ls="--", color="black", alpha=0.4)
ax.set_title(
    f"Projections of H2.X onto H0.2",
    fontsize=16,
)
ax.set_xlabel("Position")
ax.set_ylabel("Projection Ratio")
ax.legend(title="Head")

fig.tight_layout()

# Save figure
fig.savefig(FIG_FILEPATH_A, bbox_inches="tight")
print("Saved figure to file: ", FIG_FILEPATH_A)

#%%
fig, ax = plt.subplots(figsize=(12, 7))

sns.lineplot(
    data=df_sum,
    x="pos",
    y="projection_ratio",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax,
)
ax.axhline(0, ls="--", color="black", alpha=0.4)
ax.set_title(
    f"Projection of sum(H2.X) onto H0.2",
    fontsize=16,
)
ax.set_xlabel("Position")
ax.set_ylabel("Projection Ratio")

fig.tight_layout()

# Save figure
fig.savefig(FIG_FILEPATH_B, bbox_inches="tight")
print("Saved figure to file: ", FIG_FILEPATH_B)
