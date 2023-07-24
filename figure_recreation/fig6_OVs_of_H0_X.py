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

from transformer_lens import HookedTransformer
from jamesd_utils import scale_embeddings

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"
BLOCK = 0
MIN_TOKEN_COUNT = 100
FIG_FILEPATH = "figs/fig6_OVs_of_H0_X.jpg"


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
# # Unscaled embeddings
# W_E = model.W_E  # (d_vocab, d_model)
# W_pos = model.W_pos  # (n_ctx, d_model)

# Scaled embeddings
df_token_counts = pd.read_csv("../sanity_checks/counts_with_strings.csv")
token_ids_to_scale = df_token_counts.query("count >= @MIN_TOKEN_COUNT")["token_id"].tolist()
W_E, W_pos = scale_embeddings(model, token_ids_to_scale, device=device)
W_E = W_E[token_ids_to_scale]  # Keep only scaled embeddings

#%%
W_OV = []
for head in range(model.cfg.n_heads):
    W_OV.append(model.W_V[BLOCK, head] @ model.W_O[BLOCK, head])
W_OV = torch.stack(W_OV)  # (n_heads, d_model, d_model)

#%%
df = pd.DataFrame()

for head in range(model.cfg.n_heads):
    df_E = pd.DataFrame()
    df_E["norm"] = (W_E @ W_OV[head]).norm(dim=-1)
    df_E["head"] = head
    df_E["Embedding"] = "Token"

    df_pos = pd.DataFrame()
    df_pos["norm"] = (W_pos @ W_OV[head]).norm(dim=-1)
    df_pos["head"] = head
    df_pos["Embedding"] = "Positional"

    df = pd.concat([df, df_E, df_pos])

#%%
fig, ax = plt.subplots(2, 4, figsize=(12, 6))

xlims = [
    (-0.2, 5.2),
    (-0.2, 4),
    (-0.2, 18),
    (-0.2, 5),
    (-0.2, 6),
    (-0.2, 5.2),
    (-0.2, 5),
    (-0.2, 5),
]

for xlim, head in zip(xlims, range(model.cfg.n_heads)):
    row, col = divmod(head, 4)

    sns.histplot(
        data=df.query(f"head == {head}"),
        x="norm",
        hue="Embedding",
        common_norm=False,
        stat="density",
        ax=ax[row, col],
    )
    title_kwargs = dict(fontweight='bold') if head == 2 else {}
    ax[row, col].set_title(f"H{BLOCK}.{head}", **title_kwargs)
    ax[row, col].set_xlim(xlim)
    ax[row, col].set_xlabel("Norm")
    if not (row == 0 and col == 0):
        ax[row, col].get_legend().remove()

fig.suptitle(
    f"Histograms of norm(scaled_embedding @ W_OV) for each head in layer 0",
    fontsize=16,
)
fig.tight_layout()

#%%
# Save figure
fig.savefig(FIG_FILEPATH, bbox_inches="tight", dpi=300)
print("Saved figure to file: ", FIG_FILEPATH)
