# %% Import modules
import torch
torch.set_grad_enabled(False)
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 

import plotly.express as px
import pandas as pd
import numpy as np
import einops
import importlib
import sys
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Callable, Optional
from functools import partial

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
import plotly.graph_objects as go
import gc
import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib

from load_data import (
    get_prompts_t,
)
#%% Setup model & load data
model = HookedTransformer.from_pretrained('gelu-4l')
model.cfg.use_attn_result = True
model.to(device)

prompts_t = get_prompts_t()


# %% resample ablation
def resample_ablation(activation, hook, head):
    """activation: (batch, pos, head, dmodel)"""
    tmp = activation.clone()
    activation[:-1, :, head, :] = tmp[1:, :, head, :]
    activation[-1, :, head, :] = tmp[0, :, head, :]

    return activation

# %% calculate loss change for each head
BATCH_SIZE = 5

# Compute original loss
logits = model(prompts_t[:BATCH_SIZE, :], return_type="logits")
original_losses = model.loss_fn(logits=logits, tokens=prompts_t[:BATCH_SIZE, :], per_token=True)
del logits
gc.collect()

ablated_loss_diff_matrix = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, model.cfg.n_ctx-1))

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        ablated_logits = model.run_with_hooks(
            prompts_t[:BATCH_SIZE], 
            return_type="logits", 
            fwd_hooks=[(get_act_name('result', layer), partial(resample_ablation, head=head))]
        )
        losses_per_head = model.loss_fn(logits=ablated_logits, tokens=prompts_t[:BATCH_SIZE], per_token=True)
        del ablated_logits
        gc.collect()

        ablated_loss_diff_matrix[layer, head] = losses_per_head - original_losses

print(original_losses.mean(dim=(-1, -2)))
ori_loss = original_losses.mean(dim=(-1, -2)).item()
fig = px.imshow(
    ablated_loss_diff_matrix.mean(dim=(-1, -2)).round(decimals=2),
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
    title=f"Resample Ablation Loss Diff for each Head, average over {BATCH_SIZE} prompts and {model.cfg.n_ctx-1} positions<br>Average original loss: {ori_loss:.2f}",
    labels=dict(x="Head", y="Layer"),
    text_auto=True)

fig.show()

# %%
