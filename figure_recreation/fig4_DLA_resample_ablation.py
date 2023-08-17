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
from jamesd_utils import get_logit_diff_function

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

IPSUM = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
# IPSUM = ""

N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
FIG_A_FILEPATH = "figs/fig4a_DLA_resample_ablation.jpg"
FIG_B_FILEPATH = "figs/fig4b_DLA_resample_ablation.jpg"

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
model.cfg.use_split_qkv_input = True  # Required to have a head dimension in hook_normalized

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
    #   - patch q/k/v_input (this is pre-ln)
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
orig_logit_diffs = []
for example in examples:
    logits = model(example["text"])[0, -1]
    logit_diff = (
        logits[model.to_single_token(example["correct"])]
        - logits[model.to_single_token(example["incorrect"])]
    )
    orig_logit_diffs.append(logit_diff.item())


# %%
layer, head = 0, 2

orig_dlas = []
ra_dlas = []
for example in examples:
    orig_dla, clean_scale = get_head_DLA(model, example, layer, head)
    ra_dla = get_head_DLA_resample_ablation(model, example, layer, head, rand_prompts, clean_scale)
    orig_dlas.append(orig_dla)
    ra_dlas.append(ra_dla)

# %%
df = pd.DataFrame()
df["dla"] = orig_dlas
df["example"] = [i+1 for i in range(len(orig_dlas))]
df["type"] = "original"

tmp_df = pd.DataFrame()
for i, ra_dla in enumerate(ra_dlas):
    tmp_df["dla"] = ra_dla
    tmp_df["example"] = i+1
    tmp_df["type"] = "resample"
    df = pd.concat([df, tmp_df])

# %%
fig_a, ax_a = plt.subplots(2, 2, figsize=(12, 8))

for i in range(len(examples)):
    r, c = i // 2, i % 2
    sns.barplot(
        data=df.query(f"example == {i+1}"),
        x="type",
        y="dla",
        estimator="median",
        errorbar=("pi", 75),
        ax=ax_a[r, c],
    )
    # Plot aesthetics
    ax_a[r, c].set_title(
        f"Prompt: ... {repr(examples[i]['text'].replace(IPSUM, ''))}\n"
        f"Correct/incorrect token: {repr(examples[i]['correct'])} / {repr(examples[i]['incorrect'])}\n"
        f"Original logit difference: {orig_logit_diffs[i]:.2f}",
        fontsize=12,
    )
    ax_a[r, c].set_xlabel("")
    if c == 0:
        ax_a[r, c].set_ylabel("Direct Logit Attribution", fontsize=12)
    else:
        ax_a[r, c].set_ylabel("")
    ax_a[r, c].set_xticklabels(["Original", "Resample Ablation"], fontsize=12)

fig_a.suptitle(f"DLA of L{layer}H{head}, with/without resample ablation", fontsize=16)
fig_a.tight_layout()

# %%
attn_heads = [(3, 1), (3, 4), (3, 5), (3, 0),]

orig_dlas2 = []
ra_dlas2 = []
for example, (layer, head) in zip(examples, attn_heads):
    orig_dla, clean_scale = get_head_DLA(model, example, layer, head)
    ra_dla = get_head_DLA_resample_ablation(model, example, layer, head, rand_prompts, clean_scale)
    orig_dlas2.append(orig_dla)
    ra_dlas2.append(ra_dla)

# %%
df2 = pd.DataFrame()
df2["dla"] = orig_dlas2
df2["example"] = [i+1 for i in range(len(orig_dlas2))]
df2["type"] = "original"

tmp_df = pd.DataFrame()
for i, ra_dla in enumerate(ra_dlas2):
    tmp_df["dla"] = ra_dla
    tmp_df["example"] = i+1
    tmp_df["type"] = "resample"
    df2 = pd.concat([df2, tmp_df])

# %%
fig_b, ax_b = plt.subplots(2, 2, figsize=(12, 8))

for i in range(len(examples)):
    r, c = i // 2, i % 2
    sns.barplot(
        data=df2.query(f"example == {i+1}"),
        x="type",
        y="dla",
        estimator="median",
        errorbar=("pi", 75),
        ax=ax_b[r, c],
    )
    # Plot aesthetics
    ax_b[r, c].axhline(0, color="black", linestyle="--")
    ax_b[r, c].set_title(
        f"Head = L{attn_heads[i][0]}H{attn_heads[i][1]}\n"
        f"Prompt: ... {repr(examples[i]['text'].replace(IPSUM, ''))}\n"
        f"Correct/incorrect token: {repr(examples[i]['correct'])} / {repr(examples[i]['incorrect'])}\n"
        f"Original logit difference: {orig_logit_diffs[i]:.2f}",
        fontsize=12,
    )
    ax_b[r, c].set_xlabel("")
    if c == 0:
        ax_b[r, c].set_ylabel("Direct Logit Attribution", fontsize=12)
    else:
        ax_b[r, c].set_ylabel("")
    ax_b[r, c].set_xticklabels(["Original", "Resample Ablation"], fontsize=12)

fig_b.suptitle(f"DLAs of different heads, with/without resample ablation", fontsize=16)
fig_b.tight_layout()

# %%
fig_a.savefig(FIG_A_FILEPATH)
fig_b.savefig(FIG_B_FILEPATH)
print("Figures saved to: ", FIG_A_FILEPATH, FIG_B_FILEPATH)


# Code to find which heads have good anti-examples
# %%
# layer = 3
# for head in range(8):
#     print(f"L{layer}H{head}:")
#     for example in examples:
#         orig_dla, clean_scale = get_head_DLA(model, example, layer, head)
#         ra_dla = get_head_DLA_resample_ablation(model, example, layer, head, rand_prompts, clean_scale)
#         print("\t", example["text"].replace(IPSUM, ""))
#         print("\t", orig_dla, ra_dla)

# # cupboard L3H1
# # i went L3H4 or L3H3
# # MyClass L3H5
# # church L3H0