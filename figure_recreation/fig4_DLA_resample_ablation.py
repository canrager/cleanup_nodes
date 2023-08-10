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

from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from jamesd_utils import get_logit_diff_function

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

IPSUM = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
# IPSUM = ""

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1
FIG_FILEPATH = "figs/fig4_DLA_resample_ablation.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


# %%
rand_prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(rand_prompts, dim=0).shape == rand_prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True  # Required?

# %%
examples = [
    {
        "text": 4 * IPSUM + " It's in the cupboard, either on the top or the",
        "correct": " bottom",
        "incorrect": " top",
    },
    {
        "text": 5 * IPSUM + " I went to university at Michigan",
        "correct": " State",
        "incorrect": " University",
    },
    {
        "text": IPSUM + " class MyClass:\n\tdef",
        "correct": " __",
        "incorrect": " on",
    },
    {
        "text": 6 * IPSUM + "The church I go to is the Seventh-day Adventist",
        "correct": " Church",
        "incorrect": " Advent",
    },
]


# %%
def get_head_DLA(model, example, layer, head):
    """Only for final sequence position."""

    token_ids = model.to_tokens(example["text"])
    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])
    logit_diff_direction = (
        model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id]
    )  # (d_model,)

    _, cache = model.run_with_cache(
        token_ids,
        names_filter=lambda name: (
            name == f"blocks.{layer}.attn.hook_result" or name == "ln_final.hook_scale"
        ),
    )

    head_out = cache[f"blocks.{layer}.attn.hook_result"][0, -1, head, :]  # (d_model,)
    scale = cache["ln_final.hook_scale"][0, -1]  # (1,)

    apply_ln = lambda x: (x - x.mean(dim=-1, keepdim=True)) / scale
    head_out_ln = apply_ln(head_out)  # (d_model,)

    return (head_out_ln @ logit_diff_direction).item(), scale


def get_head_DLA_resample_ablation(model, example, layer, head, rand_prompts, clean_scale):
    """Patch the q/k/v inputs of a head and get the DLA"""

    # Prep data for patching
    token_ids = model.to_tokens(example["text"])
    token_ids = einops.repeat(token_ids, "b p -> (r b) p", r=rand_prompts.shape[0])

    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])
    logit_diff_direction = (
        model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id]
    )  # (d_model,)

    # TODO: think about how to correct for ln scale
    # Slightly different ways to patch:
    #   - patch q/k/v_input (will change the ln scale of other heads)
    #   - patch ln1.hook_normalized

    # Get activations for patching
    _, corrupted_cache = model.run_with_cache(
        rand_prompts[:, :token_ids.shape[1]],
        names_filter=lambda name: (
            name == f"blocks.{layer}.ln1.hook_normalized"  # (batch, pos, head, d_model)
            # or name == f"blocks.{layer}.ln1.hook_scale"
            # or name == f"blocks.{layer}.hook_q_input"
            # or name == f"blocks.{layer}.hook_k_input"
            # or name == f"blocks.{layer}.hook_v_input"
        ),
        device=device,
    )

    # Define patching hook function
    def patch_head(activations, hook):
        corrupted_activations = corrupted_cache[hook.name]
        activations[:, -1, head, :] = corrupted_activations[:, -1, head, :]

    # Add patching hooks and run with cache
    model.reset_hooks()
    model.add_hook(f"blocks.{layer}.ln1.hook_normalized", patch_head, level=1)
    _, patched_cache = model.run_with_cache(
        token_ids,
        names_filter=lambda name: (
            name == f"blocks.{layer}.attn.hook_result"
            # or name == "ln_final.hook_scale"
        ),
        device=device,
    )
    model.reset_hooks()

    # Get DLAs of the patched run
    # (batch, d_model)
    patched_head_out = patched_cache[f"blocks.{layer}.attn.hook_result"][:, -1, head, :]
    # scale = patched_cache["ln_final.hook_scale"][:, -1]  # (batch, 1)

    apply_ln = lambda x: (x - x.mean(dim=-1, keepdim=True)) / clean_scale
    patched_head_out_ln = apply_ln(patched_head_out)  # (batch, d_model)

    return patched_head_out_ln @ logit_diff_direction

# %%
example = examples[0]
orig_dla, clean_scale = get_head_DLA(model, example, 0, 2)
ra_dlas = get_head_DLA_resample_ablation(model, example, 0, 2, rand_prompts, clean_scale)

orig_dla, ra_dlas

# %%
for i in range(8):
    example = examples[0]
    orig_dla, clean_scale = get_head_DLA(model, example, 2, i)
    ra_dlas = get_head_DLA_resample_ablation(model, example, 2, i, rand_prompts, clean_scale)

    print(orig_dla, ra_dlas)
