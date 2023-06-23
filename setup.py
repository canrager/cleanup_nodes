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
from typing import List, Callable
from plotting import (
    get_fig_head_to_mlp_neuron,
    get_fig_head_to_mlp_neuron_by_layer
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

#%% Inpection - Dataset & Model
logits, activation_cache = model.run_with_cache(prompts_t[10])
# print("".join(model.to_str_tokens(prompts_t[10])))
# print("prediction:", model.to_str_tokens(logits.argmax(dim=-1)[0, -1]))

# %% Select neurons
thres = -0.4
selected_neurons = select_neurons_by_cosine_similarity(model, thres)
# %% Calculate and plot node-node projection
node_node_projections = calc_node_node_projection(
    model, 
    prompts_t, 
    selected_neurons,
    proj_func=projection
)
#%% Find concrete neuron-head pairs


# %%
get_fig_head_to_mlp_neuron(
    projections=node_node_projections,
    quantile=0.1,
    k=50,
    n_heads=model.cfg.n_heads,
    d_mlp=model.cfg.d_mlp
).show()


get_fig_head_to_mlp_neuron_by_layer(
    projections=node_node_projections,
    k=10,
    quantile=0.1,
    n_layers=model.cfg.n_layers,
).show()


# %% Calculate and plot node-resid projection

node_resid_projections = calc_node_resid_projection(model, prompts_t, projection)
plot_projection_node_resid(model, node_resid_projections, projection)


#%% Single Head -> Single Neuron: Projecion Distribution

def single_head_neuron_projection(
    prompts, 
    writer_layer: int, 
    writer_idx: int, 
    cleaner_layer: int, 
    cleaner_idx: int,
    return_fig: bool = False
) -> Float[Tensor, "projection_values"]:

    # Get act names
    writer_hook_name = get_act_name("result", writer_layer)
    cleaner_hook_name = get_act_name("post", cleaner_layer)

    # Run with cache on all prompts (at once?)
    _, cache = model.run_with_cache(
        prompts,
        names_filter=lambda name: name in [writer_hook_name, cleaner_hook_name]
        )

    # Get ouput per neuron from mlp_post
    full_neuron_output = get_neuron_output(cache, cleaner_layer, cleaner_idx)

    # Select specific idx
    # calc projections vectorized (is the projection function valid for this vectorized operation?)
    projections = projection(
        writer_out=cache[writer_hook_name][:, :, writer_idx, :],
        cleanup_out=full_neuron_output
    )

    if return_fig:
        proj_np = projections.flatten().cpu().numpy()
        seqpos_labels = einops.repeat(np.arange(model.cfg.n_ctx), "pos -> (pos n_prompts)", n_prompts=len(prompts))
        fig = px.histogram(
            proj_np,
            nbins=20,
            title=f"Projection of neuron {cleaner_layer}.{cleaner_idx} onto direction of head {writer_layer}.{writer_idx}\nfor {len(prompts) * model.cfg.n_ctx} seq positions",
            animation_frame=seqpos_labels,
            range_x=[proj_np.min(), proj_np.max()]
        )

        return projections, fig
    else:
        return projections


# %% Single Head-Neuron inspection
hn_proj, fig = single_head_neuron_projection(
    prompts_t[:20],
    writer_layer=1,
    writer_idx=6,
    cleaner_layer=2,
    cleaner_idx=1182,
    return_fig=True
)

fig.show()




#%%
# def plot_projection_node_node(node_node_projection_matrix: Float[Tensor, "head neuron prompt pos"], proj_func: Callable):
#     px.imshow(
#         node_node_projection_matrix.flatten(start_dim=-2).mean(dim=-1), # TODO percentile later
#         color_continuous_midpoint=0,
#         color_continuous_scale='RdBu',
#         y=[f"Head {layer}.{head}" for layer, head in all_heads],
#         x=[f"Resid {layer}.{resid.split('_')[-1]}" for layer, resid in all_resids],
#         title=f"Projection of Resid to Attn Head Output Direction using {proj_func.__name__}"
#     ).show()

# plot_projection_node_node(node_node_projections, projection)
# %%


# %%


# %%
