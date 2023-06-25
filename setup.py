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
    get_fig_head_to_mlp_neuron_by_layer,
    get_fig_head_to_selected_mlp_neuron
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
thres = -0.3
selected_neurons = select_neurons_by_cosine_similarity(model, thres, show_distribution=True)
print(len(selected_neurons))
# %% Calculate and plot node-node projection
node_node_projections = calc_node_node_projection(
    model, 
    prompts_t[:10], 
    selected_neurons,
    proj_func=projection
)
# %% Find concrete neuron-head pairs
all_heads = [
    (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]  # [head0.0, head0.1, ..., head2.7]


projection_quantile = node_node_projections.flatten(start_dim=-2).quantile(0.1, dim=-1)
thres = -0.3
selected_pairs = [
    (all_heads[head], selected_neurons[neuron])
    for head in range(projection_quantile.shape[0])
    for neuron in range(projection_quantile.shape[1])
    if projection_quantile[head, neuron] < thres
]
selected_pairs = [
    (head, neuron)
    for head, neuron in selected_pairs
    if head[0] < neuron[0] and neuron[0] != model.cfg.n_layers - 1
]
selected_pairs


# %%
get_fig_head_to_selected_mlp_neuron(
    projections=node_node_projections,
    k=50,
    quantile=0.1,
    neuron_names=selected_neurons,
    n_layers=model.cfg.n_layers,
).show()


# get_fig_head_to_mlp_neuron_by_layer(
#     projections=node_node_projections,
#     k=10,
#     quantile=0.1,
#     n_layers=model.cfg.n_layers,
# ).show()


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

# plot_projection_node_node(node_node_projections, projection
# %%

all_heads = [
    (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
] # [head0.0, head0.1, ..., head2.7]

all_resids = [
    (l, act) for l in range(model.cfg.n_layers) for act in ['resid_mid', 'resid_post']
] # [resid_mid0, resid_post0, resid_mid1, ... ,resid_post2]

def plot_projection_node_resid(node_projections, proj_func: Callable):
    px.imshow(
        node_projections.flatten(start_dim=-2).mean(dim=-1),
        color_continuous_midpoint=0,
        color_continuous_scale='RdBu',
        y=[f"Head {layer}.{head}" for layer, head in all_heads],
        x=[f"Resid {layer}.{resid.split('_')[-1]}" for layer, resid in all_resids],
        title=f"Projection of Resid to Attn Head Output Direction using {proj_func.__name__}"
    ).show()

node_resid_projections = calc_node_resid_projection(prompts_t, projection) # [n_heads * n_layers, n_resid_stages * n_layers, batch, pos]
plot_projection_node_resid(node_resid_projections, projection)

#%%
def plot_node_resid_projection(
    node_projections: Float[Tensor, "total_heads total_resids batch pos"],
    proj_func: Callable
) -> False:
    node_projections = einops.reduce(node_projections, "total_heads total_resids batch pos -> total_heads total_resids", "mean").flatten().cpu().numpy()
    # node_projections = einops.rearrange(
    #     node_resid_projections, 
    #     "(n_layers n_heads) total_resids -> n_layers n_heads total_resids", 
    #     n_layers=model.cfg.n_layers,
    #     n_heads=model.cfg.n_heads).flatten().cpu().numpy()
    
    nlayer_labels = einops.repeat(np.arange(model.cfg.n_layers), "layer -> (layer nheads total_resids)", nheads=model.cfg.n_heads, total_resids=8)
    nheads_labels = einops.repeat(np.arange(model.cfg.n_heads), "nheads -> (layer nheads total_resids)", layer=model.cfg.n_layers, total_resids=8)
    nresid_labels = einops.repeat(np.arange(2 * model.cfg.n_layers), "total_resids -> (layer nheads total_resids)", layer=model.cfg.n_layers, nheads=model.cfg.n_heads)

    df = pd.DataFrame({
        "n_layers": nlayer_labels,
        "n_heads": nheads_labels,
        "n_resids": nresid_labels,
        "projection_values": node_projections
    })
    fig = px.line(
            df,
            x="n_resids",
            y="projection_values",
            color="n_heads",
            title=f"",
            animation_frame="n_layers",
            range_y=[node_projections.min(), node_projections.max()]
        )

    return fig

fig = plot_node_resid_projection(node_resid_projections, projection)
fig.show()


# %% Inspect usage of direction of single writer node's output across residual stream

def single_head_full_resid_projection(
    prompts, 
    writer_layer: int, 
    writer_idx: int, 
    neuron_layer: int,
    neuron_idx: int,
    return_fig: bool = False,
) -> Float[Tensor, "projection_values"]:

    # Get act names
    writer_hook_name = get_act_name("result", writer_layer)
    resid_hook_names = ["resid_mid", "resid_post"]
    

    # Run with cache on all prompts (at once?)
    _, cache = model.run_with_cache(
        prompts,
        names_filter= lambda name: (any(hook_name in name for hook_name in resid_hook_names)) or (name == writer_hook_name)
        )
    
    # calc projections vectorized (is the projection function valid for this vectorized operation?)
    # [ n_resid = 2*n_layers, batch, pos ]

    projections = torch.zeros(size=(2*model.cfg.n_layers, len(prompts), model.cfg.n_ctx))
    for layer in range(model.cfg.n_layers):
        for i, resid_stage in enumerate(resid_hook_names):
            resid_hook_name = get_act_name(resid_stage, layer)

            projections[2*layer + i] = projection(
                writer_out=cache[writer_hook_name][:, :, writer_idx, :],
                cleanup_out=cache[resid_hook_name]
            )
    projections = einops.reduce(projections, "n_resid batch pos -> n_resid", "mean")

    if return_fig:
        resid_labels = []
        for i in range(model.cfg.n_layers):
            resid_labels.append(f"L{i}_resid_mid")
            resid_labels.append(f"L{i}_resid_post")

        d = {"projection_value": projections.cpu().numpy(),
             "labels": resid_labels}
        df = pd.DataFrame(d)

        fig = px.line(
            df,
            x="labels",
            y="projection_value",
            title=f"H{writer_layer}.{writer_idx} projection onto residual stream (linked with N{neuron_layer}.{neuron_idx})",
        )
        fig.add_vline(x=neuron_layer * 2, line_dash="dash", line_color="black")

        return projections, fig
    else:
        return projections

# %% Inspect single head full resid projection
for (hl, hi), (nl, ni) in selected_pairs:
    hr_proj, fig = single_head_full_resid_projection(
        prompts_t[0],
        writer_layer=hl,
        writer_idx=hi,
        neuron_layer=nl,
        neuron_idx=ni,
        return_fig=True,
    )

    fig.show()

# %%
