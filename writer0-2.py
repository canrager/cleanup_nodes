# %% Import modules
import torch
import einops
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name



import plotly.express as px
import pandas as pd
import numpy as np
from tqdm.notebook import trange, tqdm

from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Callable, Optional
from plotting import (
    get_fig_head_to_mlp_neuron,
    get_fig_head_to_mlp_neuron_by_layer,
    get_fig_head_to_selected_mlp_neuron,
    single_head_full_resid_projection
)
from load_data import get_prompts_t
from utils import (
    get_neuron_output, 
    act_filter,
    projection,
    cos_similarity,
    calc_node_node_projection,
    calc_node_resid_projection,
    select_neurons_by_cosine_similarity,
    plot_projection_node_resid,
)

#%% Setup model & load data
model = HookedTransformer.from_pretrained('gelu-4l')
model.cfg.use_attn_result = True
model.to(device)

prompts_t = get_prompts_t()

#%% Reproduce projection of resid to writer head 0.2 

# TODO add error bar!

hr_proj, fig = single_head_full_resid_projection(
    model,
    prompts_t[:30],
    writer_layer=0,
    writer_idx=2,
    return_fig=True,
    )

fig.show()
# %% Projection of 8 heads in layer 2 to 0.2 output, mean across 50 prompts, 1024 sequence positions each

BATCH_SIZE = 30

# Cache activations for writer and sender heads
writer_hook_name = get_act_name("result", 0)
cleaner_hook_name = get_act_name("result", 2)

_, cache = model.run_with_cache(
        prompts_t[:BATCH_SIZE],
        names_filter= lambda name: name in [writer_hook_name, cleaner_hook_name]
        )

# Calc projections
writer_direction = cache[writer_hook_name]
cleaner_directions = cache[cleaner_hook_name]

projections = np.zeros((model.cfg.n_heads, BATCH_SIZE, model.cfg.n_ctx))
for i in range(model.cfg.n_heads):
    projections[i, :, :] = projection(writer_direction[:, :, 2, :], cleaner_directions[:, :, i, :]).cpu()

# Plot MEANprojections
df = pd.DataFrame({
    "projection_values": projections.mean(axis=(1,2)),
    "cleanup_heads": np.arange(model.cfg.n_heads)
})
px.bar(df, x="cleanup_heads", y="projection_values", title=f"Projection layer 2 heads to head 0.2 output<br>mean across {BATCH_SIZE} prompts, 1024 sequence positions each")

# %% Zoom in: For each head: prompt by seq position
# projections (nheads, batch, pos)

cleaner_head_idx = 0
plot_slice = 2**2

for i in range(plot_slice):
    seq_window_len = 1024 // plot_slice
    proj = projections[cleaner_head_idx, :, i*seq_window_len:(i+1)*seq_window_len]
    px.imshow(
        proj,
        color_continuous_scale="RdBu",
        zmin=-abs(projections[cleaner_head_idx]).max(),
        zmax=abs(projections[cleaner_head_idx]).max(),
        title=f"Cleaner head2.{cleaner_head_idx} projected onto head0.2<br>seqence positions {i*seq_window_len} to {(i+1)*seq_window_len}",
        labels={
            "values": "layer 2 heads"
        }
    ).show()



# %% Compare cleaner heads

cleaner_head_idx = 6
plot_slice = 2**2

for i in range(plot_slice):
    seq_window_len = 1024 // plot_slice
    proj = projections[cleaner_head_idx, :, i*seq_window_len:(i+1)*seq_window_len]
    px.imshow(
        proj,
        color_continuous_scale="RdBu",
        zmin=-abs(projections[cleaner_head_idx]).max(),
        zmax=abs(projections[cleaner_head_idx]).max(),
        title=f"Cleaner head2.{cleaner_head_idx} projected onto head0.2<br>seqence positions {i*seq_window_len} to {(i+1)*seq_window_len}"
    ).show()

#%% 