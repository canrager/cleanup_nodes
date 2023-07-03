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

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1


#%%
def get_bos_values(
    attn_pattern: Float[Tensor, "... posQ posK"],
) -> Float[Tensor, "... bos_values"]:
    assert attn_pattern.shape[-2] == attn_pattern.shape[-1]

    return attn_pattern[..., :, 0]


def get_nonbos_values(
    attn_pattern: Float[Tensor, "... posQ posK"],
) -> Float[Tensor, "... nonbos_values"]:
    assert attn_pattern.shape[-2] == attn_pattern.shape[-1]
    dim_neg2, dim_neg1 = torch.tril_indices(
        attn_pattern.shape[-1] - 1, attn_pattern.shape[-1] - 1,
    ).unbind()

    return attn_pattern[..., 1:, 1:][..., dim_neg2, dim_neg1]


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

# ------------------ ATTENTION PATTERN HEATMAP PLOTS ------------------------ #
#%%
prompts_text = [
    "argaera oaejopirja oaijgro sijaosaig oiargoas ijsori gaoirs",
    "aergp aergaRAG apiej epira ejoerigajer iaeo joaeirg oijear",
    "otaijpsd7g piaujr[oeaim [oaia[ oia oiu pgaoeirao ih i]]]",
    "argaera oaejopirja oaijgro sijaosaig oiargoas ijsori gaoirs",
    "aergp aergaRAG apiej epira ejoerigajer iaeo joaeirg oijear",
    "argaera oaejopirja oaijgro sijaosaig oiargoas ijsori gaoirs",
    "aergp aergaRAG apiej epira ejoerigajer iaeo joaeirg oijear",
    "otaijpsd7g piaujr[oeaim [oaia[ oia oiu pgaoeirao ih i]]]",
]

prompts_rand = torch.tensor(
    model.tokenizer(prompts_text, padding=True)["input_ids"],
    device=device,
)[:, :20]

_, cache = model.run_with_cache(
    prompts_rand,
    names_filter=lambda name: "blocks.0.attn.hook_pattern" in name,
    device=device,
)

attn_patterns = cache["blocks.0.attn.hook_pattern"][:, 2, :, :]
"Shape: (batch, posQ, posK)"

#%%
# Plot attention patterns for some random prompts
fig_a, ax_a = plt.subplots(2, 4, figsize=(14, 6))

for i in range(attn_patterns.shape[0]):
    row, col = divmod(i, 4)

    sns.heatmap(
        attn_patterns[i],
        vmin=0,
        vmax=1,
        cmap="Blues",
        ax=ax_a[row, col],
        cbar=False,
    )
    ax_a[row, col].set_title(f"Prompt {i+1}")
    ax_a[row, col].set_xticks([])
    ax_a[row, col].set_yticks([])

fig_a.tight_layout()

#%%
# Save figure
fig_a.savefig(FIG_A_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_A_FILEPATH)

# -------------------- ATTENTION PATTERN LINE PLOTS ------------------------- #

#%%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

#%%
_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: "blocks.0.attn.hook_pattern" in name,
    device=device,
)

#%%
attn_patterns = cache["blocks.0.attn.hook_pattern"][:, 2, :, :]
"Shape: (batch, posQ, posK)"

bos_values = get_bos_values(attn_patterns)
nonbos_values = get_nonbos_values(attn_patterns)

df_bos = ntensor_to_long(
    bos_values,
    value_name="attn_prob",
    dim_names=["batch", "pos"],
)

df_nonbos = ntensor_to_long(
    nonbos_values,
    value_name="attn_prob",
    dim_names=["batch", "nonbos_pos"],
)

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
    f"Attention Probabilities in H0.2\n"
    f"Batch size: {prompts.shape[0]}, error bars: q25-q75"
)

# Subplot 2
sns.lineplot(
    data=df_nonbos.query("nonbos_pos <= 2000"),
    x="nonbos_pos",
    y="attn_prob",
    estimator="median",
    errorbar=("pi", 75),
    ax=ax_b[1],
)
ax_b[1].set_xlabel("Position of Flattened Lower Triangular without BOS Column")
ax_b[1].set_ylabel("Attention Probability")
ax_b[1].set_title(
    f"Attention Probabilities when Attending to non-BOS in H0.2\n"
    f"Batch size: {prompts.shape[0]}, error bars: q25-q75"
)

fig_b.tight_layout()

#%%
# Save figure
fig_b.savefig(FIG_B_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_B_FILEPATH)


