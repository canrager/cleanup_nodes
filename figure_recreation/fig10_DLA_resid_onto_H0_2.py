#%%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path
parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)
    
# %% Import modules
import torch
torch.set_grad_enabled(False)
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 

import plotly.express as px
import pandas as pd
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Callable, Optional, Tuple

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name, test_prompt
import plotly.graph_objects as go

import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib

from jamesd_utils import (
    projection_ratio
)
FIGS_FILEPATH = [
    "figs/fig10a_DLA_resid_onto_H0_2.jpg",
    "figs/fig10b_DLA_resid_onto_H0_2.jpg",
    "figs/fig10c_DLA_resid_onto_H0_2.jpg",
]

#%% Setup model & load data
model = HookedTransformer.from_pretrained('gelu-4l')
model.cfg.use_attn_result = True
model.to(device)
# %%
examples = [
    {'text': "It's in the shelf, either on the top or the", 'answer': ' bottom'},
    {'text': "I went to university at Michigan", 'answer': ' State'},
    {'text': "class MyClass:\n\tdef", 'answer': ' __'},
]

# %%
def plot_H02_to_resid(prompt, answer, model):
    logits, sample_cache = model.run_with_cache(prompt)
    next_token = logits.argmax(dim=-1)[..., -1]
    next_token = model.tokenizer.decode(next_token)

    attn_out_H02: Float[Tensor, 'dmodel'] = sample_cache['result', 0][0, -1, 2, :]

    # prepare resids for each location
    resids = torch.zeros(model.cfg.n_layers * 2 + 1, model.cfg.d_model)

    # resid_pre_0
    resids[0] = sample_cache['resid_pre', 0][0, -1, :]

    # resid mid and post
    for layer in range(model.cfg.n_layers):
        resids[2 * layer + 1] = sample_cache['resid_mid', layer][0, -1, :]
        resids[2 * layer + 2] = sample_cache['resid_post', layer][0, -1, :]

    # calculate projection ratio
    resid_onto_attn_out = torch.zeros(model.cfg.n_layers * 2 + 1)
    for i in range(model.cfg.n_layers * 2 + 1):
        resid_onto_attn_out[i] = projection_ratio(resids[i], attn_out_H02)

    xticks = ['resid_pre_0'] + [f'resid_{act}_{layer}' for layer in range(model.cfg.n_layers) for act in ['mid', 'post']]
    
    return px.line(
        y=resid_onto_attn_out, 
        x=xticks, 
        title=f'Project various residual stream location onto H0.2 at the last position<br>Prompt: "{prompt}", Next token: "{answer}"',
        labels={'x': 'residual layer', 'y': 'projection ratio'}
    )

for i, fig_path in enumerate(FIGS_FILEPATH):
    fig = plot_H02_to_resid(examples[i]['text'], examples[i]['answer'], model)
    fig.write_image(fig_path)

# %%
