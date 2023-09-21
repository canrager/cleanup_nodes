# %%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path

parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import torch
import einops
import pandas as pd

from tqdm.auto import trange
from transformer_lens import HookedTransformer
from load_data import get_prompts_t

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
SEED = 5235
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cpu"

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
# Adversarial prompts
examples = [
    {
        "text": "It's in the cupboard, either on the top or on the",
        "correct": " bottom",
        "incorrect": " top",
        # Logit diff = 1.07
    },
    {
        "text": "I went to university at Michigan",
        "correct": " State",
        "incorrect": " University",
        # Logit diff = 1.89
    },
    {
        "text": "class MyClass:\n\tdef",
        "correct": " __",
        "incorrect": " get",
        # Logit diff = 3.02
    },
    {
        "text": "The church I go to is the Seventh-day Adventist",
        "correct": " Church",
        "incorrect": " church",
        # Logit diff = 0.94
    },
]

# %%
torch.manual_seed(42)
batch_idx_shfl = torch.randperm(rand_prompts.shape[0])

# %%
# Helper functions for DLA
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
            name == f"blocks.{layer}.attn.hook_result"
            or name == "ln_final.hook_scale"
        ),
    )

    head_out = cache[f"blocks.{layer}.attn.hook_result"][0, -1, head, :]  # (d_model,)
    scale = cache["ln_final.hook_scale"][0, -1]  # (1,)

    apply_ln = lambda x: (x - x.mean(dim=-1, keepdim=True)) / scale
    head_out_ln = apply_ln(head_out)  # (d_model,)

    # Need the scale of the clean run to use for layernoming in the resample ablation run
    return (head_out_ln @ logit_diff_direction).item(), scale


def get_head_DLA_resample_ablation(model, example, layer, head, rand_prompts, clean_scale):
    """Patch the post-layernorm input of a head and get the DLA"""

    # Prep data for patching
    token_ids = model.to_tokens(example["text"])
    token_ids = einops.repeat(token_ids, "b p -> (r b) p", r=rand_prompts.shape[0])

    correct_token_id = model.to_single_token(example["correct"])
    incorrect_token_id = model.to_single_token(example["incorrect"])
    logit_diff_direction = (
        model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id]
    )  # (d_model,)

    # Get activations for patching
    _, corrupted_cache = model.run_with_cache(
        rand_prompts[:, :token_ids.shape[1]],
        names_filter=lambda name: (
            name == f"blocks.{layer}.ln1.hook_normalized"  # (batch, pos, head, d_model)
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

    apply_ln = lambda x: (x - x.mean(dim=-1, keepdim=True)) / clean_scale
    patched_head_out_ln = apply_ln(patched_head_out)  # (batch, d_model)

    return patched_head_out_ln @ logit_diff_direction

# %%
# Compute original logit diffs
orig_logit_diffs = []
for example in examples:
    logits = model(example["text"])[0, -1]
    logit_diff = (
        logits[model.to_single_token(example["correct"])]
        - logits[model.to_single_token(example["incorrect"])]
    )
    orig_logit_diffs.append(logit_diff.item())


# Get vanilla DLAs and resample ablated DLAs for the writer head
layer, head = 0, 2

orig_dlas = []
ra_dlas = []
for example in examples:
    orig_dla, clean_scale = get_head_DLA(model, example, layer, head)
    ra_dla = get_head_DLA_resample_ablation(
        model, example, layer, head, rand_prompts[batch_idx_shfl], clean_scale,
    )
    orig_dlas.append(orig_dla)
    ra_dlas.append(ra_dla)

# Prep df for plotting
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
# Get vanilla DLAs and resample ablated DLAs for the heads with meaningful
# contributions to final logits
attn_heads = [(3, 1), (2, 1), (3, 5), (2, 2)]

orig_dlas2 = []
ra_dlas2 = []
for i in trange(len(examples)):
    example = examples[i]
    layer_, head_ = attn_heads[i]

    orig_dla, clean_scale = get_head_DLA(model, example, layer_, head_)
    ra_dla = get_head_DLA_resample_ablation(
        model, example, layer_, head_, rand_prompts[batch_idx_shfl], clean_scale,
    )
    orig_dlas2.append(orig_dla)
    ra_dlas2.append(ra_dla)

# Prep df for plotting
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
fig_a, ax_a = plt.subplots(1, 4, figsize=(10, 4))

for i, ax in enumerate(ax_a):
    sns.barplot(
        data=df.query(f"example == {i+1}"),
        x="type",
        y="dla",
        estimator="median",
        errorbar=("pi", 75),
        ax=ax,
    )
    # Plot aesthetics
    ax.set_title(f"Prompt {i+1}", fontsize=14)
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("logit difference", fontsize=14)
    else:
        ax.set_ylabel("")
    ax.set_xticklabels(["clean", "patched"], fontsize=14)

fig_a.suptitle(f"DLA of L{layer}H{head}, logit difference of top 2 predictions", fontsize=16)
fig_a.tight_layout()

# %%
fig_b, ax_b = plt.subplots(1, 4, figsize=(10, 4))

for i, ax in enumerate(ax_b):
    sns.barplot(
        data=df2.query(f"example == {i+1}"),
        x="type",
        y="dla",
        estimator="median",
        errorbar=("pi", 75),
        ax=ax,
    )
    # Plot aesthetics
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(
        f"Prompt {i+1}, L{attn_heads[i][0]}H{attn_heads[i][1]}",
        fontsize=14
    )
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("logit difference", fontsize=14)
    else:
        ax.set_ylabel("")
    ax.set_xticklabels(["clean", "patched"], fontsize=14)

fig_b.suptitle(f"DLA of other heads, logit difference of top 2 predictions", fontsize=16)
fig_b.tight_layout()

# %%
fig_a.savefig(FIG_A_FILEPATH)
fig_b.savefig(FIG_B_FILEPATH)
print("Figures saved to: ", FIG_A_FILEPATH, FIG_B_FILEPATH)


# Code to find which heads have good anti-examples
# %%
# layer = 2
# for head in trange(8):
#     print(f"L{layer}H{head}:")
#     # for example in examples:
#     example = examples[3]
#     orig_dla, clean_scale = get_head_DLA(model, example, layer, head)
#     ra_dla = get_head_DLA_resample_ablation(model, example, layer, head, rand_prompts, clean_scale)
#     print("  ", repr(example["text"]))
#     print("  ", orig_dla, ra_dla)

# # cupboard : L3H1, L3H5
# # i went   : L2H1, L3H4, L3H3
# # MyClass  : L3H5, L3H0
# # church   : L2H2, L3H1