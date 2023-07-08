# %%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path

parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# %% Import modules
import torch

torch.set_grad_enabled(False)
device = torch.device("cpu")  #'cuda' if torch.cuda.is_available() else

import plotly.express as px
import pandas as pd
import numpy as np
import einops
import importlib
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Callable, Optional
from functools import partial
from tqdm.auto import tqdm
from itertools import product

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
import plotly.graph_objects as go
import gc
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()  # Setting seaborn as default style even if use only matplotlib

from load_data import (
    get_prompts_t,
)

N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
FIG_FILEPATH = "figs/fig7_resample_ablation.jpg"

BATCH_SIZE = N_TEXT_PROMPTS + N_CODE_PROMPTS
MINI_BATCH_SIZE = 5

# %% Setup model & load data
model = HookedTransformer.from_pretrained("gelu-4l")
model.cfg.use_attn_result = True
model.to(device)

# prompts_t = get_prompts_t()
prompts_t = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)


# %% resample ablation
def resample_ablation(activation, hook, head):
    """activation: (batch, pos, head, dmodel)"""
    tmp = activation.clone()
    activation[:-1, :, head, :] = tmp[1:, :, head, :]
    activation[-1, :, head, :] = tmp[0, :, head, :]

    return activation

# Compute loss diff using minibatch
def compute_loss_diff_minibatch(prompts_t, model):
    logits = model(prompts_t, return_type="logits")
    ori_loss: Float[Tensor, "batch"] = model.loss_fn(
        logits=logits, tokens=prompts_t, per_token=True
    ).mean(dim=-1)

    ablated_loss_diff = torch.zeros(
        (model.cfg.n_layers, model.cfg.n_heads, prompts_t.shape[0])
    )

    progress_bar = tqdm(
        product(range(model.cfg.n_layers), range(model.cfg.n_heads)),
        total=model.cfg.n_layers * model.cfg.n_heads,
    )

    for layer, head in progress_bar:
        ablated_logits = model.run_with_hooks(
            prompts_t,
            return_type="logits",
            fwd_hooks=[
                (get_act_name("result", layer), partial(resample_ablation, head=head))
            ],
        )
        losses_per_head: Float[Tensor, "batch"] = model.loss_fn(
            logits=ablated_logits, tokens=prompts_t, per_token=True
        ).mean(dim=-1)

        ablated_loss_diff[layer, head] = losses_per_head - ori_loss

    return ablated_loss_diff, ori_loss


def compute_loss_diff(prompts_t, model, batch_size, mb_size=5):
    ablated_loss_diff = torch.zeros(
        (model.cfg.n_layers, model.cfg.n_heads, batch_size)
    )
    ori_loss = torch.zeros(batch_size)
    for i in tqdm(range(0, batch_size, mb_size)):
        (
            ablated_loss_diff[:, :, i : i + mb_size],
            ori_loss[i : i + mb_size],
        ) = compute_loss_diff_minibatch(
            prompts_t[i : i + mb_size], model
        )
    return ablated_loss_diff, ori_loss

# %% calculate loss change for each head
ablated_loss_diff, ori_loss = compute_loss_diff(prompts_t, model, BATCH_SIZE, MINI_BATCH_SIZE)
print("Done computing loss diff")
print(f"Original losses: {ori_loss.mean().item()}")
# %% generate plot

fig = px.imshow(
    ablated_loss_diff.mean(dim=-1).round(decimals=2),
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
    title=f"Resample Ablation Loss Diff for each Head, average over {BATCH_SIZE} prompts and {model.cfg.n_ctx-1} positions<br>Average original loss: {ori_loss.mean().item():.2f}",
    labels=dict(x="Head", y="Layer"),
    text_auto=True,
)

# %%
# Write the figure to file
fig.write_image(FIG_FILEPATH)  # requires kaleido installed
print("Saved figure to file: ", FIG_FILEPATH)
# %%
