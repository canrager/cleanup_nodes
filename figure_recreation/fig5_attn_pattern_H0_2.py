#%%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path
parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import pandas as pd
import torch
import einops

from jaxtyping import Float
from torch import Tensor

from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from plotting import ntensor_to_long

import matplotlib.pyplot as plt
import seaborn as sns

# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

FIG_A_FILEPATH = "figs/fig5a_attn_pattern_H0_2.jpg"
FIG_B_FILEPATH = "figs/fig5b_attn_pattern_H0_2.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"
N_TOKENS_FOR_HEATMAP = 20

N_TEXT_PROMPTS = 7
N_CODE_PROMPTS = 1


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

# ------------------ ATTENTION PATTERN HEATMAP PLOTS ------------------------ #
#%%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

_, cache = model.run_with_cache(
    prompts[:8, :N_TOKENS_FOR_HEATMAP],
    # prompts_rand,
    names_filter=lambda name: "blocks.0.attn.hook_pattern" in name,
    device=device,
)

attn_patterns = cache["blocks.0.attn.hook_pattern"][:, 2, :, :]
"Shape: (batch, posQ, posK)"

#%%
# Plot attention patterns for some random prompts
fig_a, ax_a = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)

for i in range(attn_patterns.shape[0]):
    row, col = divmod(i, 4)

    sns.heatmap(
        attn_patterns[i],
        vmin=0,
        vmax=1,
        cmap="Blues",
        ax=ax_a[row, col],
        cbar=col == 3,
    )
    ax_a[row, col].set_title(f"Prompt {i+1}")
    if row == 1:
        ax_a[row, col].set_xlabel("Key Position (Src)")
    if col == 0:
        ax_a[row, col].set_ylabel("Query Position (Dest)")
    ax_a[row, col].set_xticks([])
    ax_a[row, col].set_yticks([])

fig_a.suptitle(
    f"Attention Patterns of H0.2 from Randomly Sampled Prompts (First {N_TOKENS_FOR_HEATMAP} Positions)"
)
fig_a.tight_layout()

#%%
# Save figure
fig_a.savefig(FIG_A_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_A_FILEPATH)

# -------------------- ATTENTION PATTERN LINE PLOTS ------------------------- #
#%%
_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: "blocks.0.attn.hook_pattern" in name,
    device=device,
)

#%%
attn_patterns = cache["blocks.0.attn.hook_pattern"][:, 2, :, :]
"Shape: (batch, posQ, posK)"

df_bos = ntensor_to_long(
    attn_patterns[..., :, 0],
    value_name="attn_prob",
    dim_names=["batch", "pos"],
)

df_nonbos = ntensor_to_long(
    attn_patterns[..., :, 1:],
    value_name="attn_prob",
    dim_names=["batch", "pos", "posK"],
)
df_nonbos = df_nonbos[df_nonbos.attn_prob > 1e-8]  # Filter out 0s

#%%
fig_b, ax_b = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1
sns.lineplot(
    data=df_bos,
    x="pos",
    y="attn_prob",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax_b[0],
)
ax_b[0].set_xlabel("Query Position of BOS")
ax_b[0].set_ylabel("Attention Probability")
ax_b[0].set_title(
    f"Attention Probabilities in H0.2 of the BOS Column\n"
    f"Median across batch (n={prompts.shape[0]})\n"
    f"Error bars: q25-q75"
)

# Subplot 2
sns.lineplot(
    data=df_nonbos,
    x="pos",
    y="attn_prob",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax_b[1],
)
ax_b[1].set_xlabel("Query Position of non-BOS")
ax_b[1].set_ylabel("Attention Probability")
ax_b[1].set_title(
    f"Attention Probabilities in H0.2 of the non-BOS Values\n"
    f"Median across batch (n={prompts.shape[0]}) and key position\n"
    f"The ith query position has i non-BOS values along the key axis\n"
    f"Error bars: q25-q75"
)

fig_b.tight_layout()

#%%
# Save figure
fig_b.savefig(FIG_B_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_B_FILEPATH)
