# %%
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

from tqdm.auto import trange
from transformer_lens import HookedTransformer
from load_data import get_prompts_t

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"


N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
FIG_FILEPATH = "figs/fig5_DLA_scatter_plot.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


# %%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(prompts, dim=0).shape == prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

# %%
def get_dla(model, prompt_token_ids, start_pos=0):
    # Get activations
    _, cache = model.run_with_cache(
        prompt_token_ids,
        names_filter=lambda name: (
            name == "blocks.0.attn.hook_result"
            or name == "blocks.2.ln1.hook_scale"
            or name == "blocks.2.attn.hook_result"
            or name == "ln_final.hook_scale"
        ),
    )

    writer_output = cache["blocks.0.attn.hook_result"][:, :, 2, :]  # (batch, pos, d_model)
    writer_block2_ln = (writer_output - writer_output.mean(dim=-1, keepdim=True)) / cache["blocks.2.ln1.hook_scale"]

    # Get V-composition of L0H2 to L2HX
    # Use six heads only
    L2HX_W_OV = einops.einsum(
        model.W_V[2, 2:],
        model.W_O[2, 2:],
        "head d_model_V d_head, head d_head d_model_O -> head d_model_V d_model_O"
    )

    v_comp = einops.einsum(
        writer_block2_ln, 
        L2HX_W_OV,
        "batch pos d_model_V, head d_model_V d_model_O -> batch pos d_model_O"
    )

    # Apply final layer norm scale
    scale = cache["ln_final.hook_scale"]
    apply_ln = lambda x: (x - x.mean(dim=-1, keepdim=True)) / scale
    writer_ln = apply_ln(writer_output)[:, start_pos:-1, :]  # There's no correct prediction for the last token
    v_comp_ln = apply_ln(v_comp)[:, start_pos:-1, :]

    # Retrieve correct next tokens from prompt
    correct_token_ids = prompt_token_ids[:, start_pos+1:]  # (batch pos)

    # Flatten batch dim 
    # (batch has to be flattened anyways for scatterplot, flattening now bc accessing model.W_U with single flattened token array is easier)
    writer_ln = einops.rearrange(
        writer_ln,
        "batch pos d_model -> (batch pos) d_model"
    )

    v_comp_ln = einops.rearrange(
        v_comp_ln,
        "batch pos d_model -> (batch pos) d_model"
    )
    correct_token_ids = einops.rearrange(
        correct_token_ids,
        "batch pos -> (batch pos)"
    )

    # Dot prod with unembed vector corresponding to correct next token
    correct_logit_directions = model.W_U[:, correct_token_ids] # (d_model pos)

    writer_dla = einops.einsum(
        writer_ln, # last token has correct prediction in prompt
        correct_logit_directions,
        "pos d_model, d_model pos-> pos"
    )

    v_comp_dla = einops.einsum(
        v_comp_ln,
        correct_logit_directions,
        "pos d_model, d_model pos -> pos"
    )

    # return writer_dla, cleaner_dla, v_comp_dla
    return writer_dla, v_comp_dla

# %%
n_prompts = 10
start_pos = 32

writer_dla, v_comp_dla = get_dla(model, prompts[:n_prompts], start_pos=start_pos)

# %%
fig, ax = plt.subplots(figsize=(7, 6))

sns.scatterplot(
    x=writer_dla,
    y=v_comp_dla,
    alpha=0.4,
    ax=ax,
)
line = torch.linspace(start=min(writer_dla), end=max(writer_dla), steps=20)
ax.plot(line, -line, color='black')

ax.set_title(
    f"DLA of predicting next token in prompt\nStarting at pos={start_pos}, n={n_prompts}",
    fontsize=14,
)
ax.set_xlabel("DLA of writer output", fontsize=13)
ax.set_ylabel("DLA of V-composition (writer->cleaner)", fontsize=13)

fig.tight_layout();

# %%
fig.savefig(FIG_FILEPATH)
print("Figure saved to: ", FIG_FILEPATH)
