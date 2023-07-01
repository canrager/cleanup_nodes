#%%
import pandas as pd
import torch
import einops

from transformer_lens import HookedTransformer

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

FIG_FILEPATH = "figs/fig5_OVs_of_H0_X.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"
BLOCK = 0


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
W_E = model.W_E  # (d_vocab, d_model)
W_pos = model.W_pos  # (n_ctx, d_model)

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
fig, ax = plt.subplots(4, 2, figsize=(10, 12), sharex=True)

for head in range(model.cfg.n_heads):
    row, col = divmod(head, 2)

    sns.kdeplot(
        data=df.query(f"head == {head}"),
        x="norm",
        hue="Embedding",
        common_norm=False,
        fill=True,
        ax=ax[row, col],
    )
    ax[row, col].set_title(f"Embedding @ H{BLOCK}.{head} @ W_OV")
    ax[row, col].set_xlim([-0.2, 4.2])
    ax[row, col].set_xlabel("Norm")

fig.tight_layout()

#%%
# Save figure
fig.savefig(FIG_FILEPATH, bbox_inches="tight", dpi=300)
print("Saved figure to file: ", FIG_FILEPATH)
