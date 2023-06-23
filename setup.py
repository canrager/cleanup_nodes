# %% Import modules
import torch
import einops
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformer_lens import HookedTransformer, ActivationCache
import plotly.express as px
from tqdm.notebook import trange, tqdm

from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Callable
from plotting import get_fig_head_to_mlp_neuron
from load_data import get_prompts_t

#%% Setup model & load data
model = HookedTransformer.from_pretrained('gelu-4l')
model.cfg.use_attn_result = True
model.to(device)

prompts_t = get_prompts_t()


#%% Inpection - Dataset & Model

logits, activation_cache = model.run_with_cache(prompts_t[10])
# print("".join(model.to_str_tokens(prompts_t[10])))
# print("prediction:", model.to_str_tokens(logits.argmax(dim=-1)[0, -1]))

# %% Implement get_neuron_output function, get neuron output for a specific neuron

def get_neuron_output(
    cache: ActivationCache, layer_idx: int, neuron_idx: int
) -> Float[Tensor, "batch pos dmodel"]:
    neuron_activation = cache["mlp_post", layer_idx][:, :, neuron_idx]
    neuron_wout = model.W_out[layer_idx, neuron_idx]
    mlp_bias = model.b_out[layer_idx] / model.cfg.d_mlp         # TODO design choice, discuss with Jett

    neuron_out = neuron_activation.unsqueeze(dim=-1) * neuron_wout + mlp_bias
    return neuron_out

# not using now
def get_full_layer_neuron_output(
    cache: ActivationCache, layer_idx: int
) -> Float[Tensor, "batch pos dmlp dmodel"]:
    neuron_activation = cache["mlp_post", layer_idx] # batch, pos, dmlp
    neuron_wout = model.W_out[layer_idx] # dmlp, dmodel
    mlp_bias = model.b_out[layer_idx] / model.cfg.d_mlp # dmodel

    neuron_out = einops.einsum(
        neuron_activation, 
        neuron_wout,
        "batch pos dmlp, dmlp dmodel -> batch pos dmlp dmodel",
    )
    del neuron_activation
    return neuron_out + mlp_bias


# test our get_neuron_output function
# custom_mlp_out = sum(
#     [get_neuron_output(activation_cache, layer, neuron_idx) for neuron_idx in range(model.cfg.d_mlp)]
# )

# torch.isclose(custom_mlp_out, activation_cache["mlp_out", layer], atol= 1e-5).all()

# %% Get pattern with transformerlens
def act_filter(name: str) -> bool:
    hook_names = ["result", "post", "resid_mid", "resid_post"]
    return any(hook_name in name for hook_name in hook_names)


# %% Projection functions
def projection(
    writer_out: Float[Tensor, 'batch pos dmodel'], 
    cleanup_out: Float[Tensor, 'batch pos dmodel']
) -> Float[Tensor, 'batch pos']:
    """Compute the projection from the cleanup output vector to the writer output direction"""
    norm_writer_out = torch.norm(writer_out, dim=-1, keepdim=True)
    dot_prod = einops.einsum(
        writer_out / norm_writer_out, 
        cleanup_out, 
        "batch pos dmodel, batch pos dmodel -> batch pos"
    )
    return dot_prod 

def cos_similarity(
    writer_out: Float[Tensor, 'batch pos dmodel'], 
    cleanup_out: Float[Tensor, 'batch pos dmodel']
) -> Float[Tensor, 'batch pos']:
    """Compute the projection from the cleanup output vector to the writer output direction"""
    norm_writer_out = torch.norm(writer_out, dim=-1, keepdim=True)
    norm_cleaner_out = torch.norm(cleanup_out, dim=-1, keepdim=True)
    dot_prod = einops.einsum(
        writer_out / norm_writer_out, 
        cleanup_out / norm_cleaner_out, 
        "batch pos dmodel, batch pos dmodel -> batch pos"
    )
    return dot_prod

def projection_full_layer_neurons(
    writer_out: Float[Tensor, 'batch pos dmodel'], 
    cleanup_out: Float[Tensor, 'batch pos dmlp dmodel']
) -> Float[Tensor, 'dmlp batch pos']:
    """Compute the projection from the cleanup output vector to the writer output direction"""
    norm_writer_out = torch.norm(writer_out, dim=-1, keepdim=True)
    dot_prod = einops.einsum(
        writer_out / norm_writer_out, 
        cleanup_out, 
        "batch pos dmodel, batch pos dmlp dmodel -> dmlp batch pos"
    )
    del writer_out
    del cleanup_out
    return dot_prod 

# %%
def calc_node_node_projection_full_mlp(
    tokens: List[int],
    proj_func: Callable,
    mb_size: int = 1,
) -> Float[Tensor, "head neuron prompt pos"]:
    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ] # [head0.0, head0.1, ..., head2.7]

    all_neurons = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.d_mlp)
    ] # [neuron0.0, neuron0.1, ..., neuron2.2047]

    n_total_heads = len(all_heads)
    n_total_neurons = len(all_neurons)
    n_prompts = len(tokens)
    n_pos = len(tokens[0])
    
    projection_matrix = torch.zeros(
        n_total_heads,
        n_total_neurons,
        n_prompts,
        n_pos,
    )

    for pi in trange(0, len(tokens), mb_size):
        _, cache = model.run_with_cache(
            tokens[pi:pi+mb_size], names_filter=act_filter
        )

        for h, (h_layer, h_idx) in enumerate(all_heads):
            for mlp_layer in range(model.cfg.n_layers):

                writer_out = cache['result', h_layer][:, :, h_idx, :]
                cleaner_out = get_full_layer_neuron_output(cache, mlp_layer)

                n_start = mlp_layer * model.cfg.d_mlp
                n_end = (mlp_layer + 1) * model.cfg.d_mlp

                projection_matrix[h, n_start:n_end, pi:pi+10, :] = proj_func(
                    writer_out, cleaner_out
                )

    return projection_matrix

# %%
# old (and slow) version
def calc_node_node_projection(
    tokens: List[int],
    proj_func: Callable,
) -> Float[Tensor, "head neuron prompt pos"]:
    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ] # [head0.0, head0.1, ..., head2.7]

    all_neurons = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.d_mlp)
    ] # [neuron0.0, neuron0.1, ..., neuron2.2047]

    n_total_heads = len(all_heads)
    n_total_neurons = len(all_neurons)
    n_prompts = len(tokens)
    n_pos = len(tokens[0])
    
    projection_matrix = torch.zeros(
        n_total_heads,
        n_total_neurons,
        n_prompts,
        n_pos,
    )

    for pi in trange(0, len(tokens), 10):
        _, cache = model.run_with_cache(
            tokens[pi : pi + 10], names_filter=act_filter
        )

        for w, (lw, hw) in enumerate(all_heads):
            for c, (lc, hc) in enumerate(all_neurons):
                writer_out = cache['result', lw][:, :, hw, :]
                cleaner_out = get_neuron_output(cache, lc, hc)
                projection_matrix[w, c, pi:pi+10, :] = proj_func(
                    writer_out, cleaner_out
                )

    return projection_matrix
# %%

# node_node_projections = calc_node_node_projection(prompts_t[:10], proj_func=projection)
node_node_projections = calc_node_node_projection_full_mlp(prompts_t, proj_func=projection_full_layer_neurons, mb_size=1)

# %%
# torch.save(node_node_projections, "node_node_projections.pt")
#%%
get_fig_head_to_mlp_neuron(
    projections=node_node_projections,
    quantile=0.1,
    k=50,
    n_heads=model.cfg.n_heads,
    d_mlp=model.cfg.d_mlp
).show()





#%%

def plot_projection_node_node(node_node_projection_matrix: Float[Tensor, "head neuron prompt pos"], proj_func: Callable):
    px.imshow(
        node_node_projection_matrix.flatten(start_dim=-2).mean(dim=-1), # TODO percentile later
        color_continuous_midpoint=0,
        color_continuous_scale='RdBu',
        y=[f"Head {layer}.{head}" for layer, head in all_heads],
        x=[f"Resid {layer}.{resid.split('_')[-1]}" for layer, resid in all_resids],
        title=f"Projection of Resid to Attn Head Output Direction using {proj_func.__name__}"
    ).show()

plot_projection_node_node(node_node_projections, projection)
# %%

def calc_node_resid_projection(
    tokens: List[int],
    proj_func: Callable,
) -> Float[Tensor, "head resid prompt pos"]:
    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ] # [head0.0, head0.1, ..., head2.7]

    all_resids = [
        (l, act) for l in range(model.cfg.n_layers) for act in ['resid_mid', 'resid_post']
    ] # [resid_mid0, resid_post0, resid_mid1, ... ,resid_post2]

    n_total_heads = len(all_heads)
    n_total_resids = len(all_resids)
    n_prompts = len(tokens)
    n_pos = len(tokens[0])
    
    projection_matrix = torch.zeros(
        n_total_heads,
        n_total_resids,
        n_prompts,
        n_pos,
    )

    for pi in trange(0, len(tokens), 10):
        _, cache = model.run_with_cache(
            tokens[pi : pi + 10], names_filter=act_filter
        )

        for w, (lw, hw) in enumerate(all_heads):
            for resid_idx, (resid_layer, resid_name) in enumerate(all_resids):
                writer_out = cache['result', lw][:, :, hw, :]
                resid_vec = cache[resid_name, resid_layer]
                projection_matrix[w, resid_idx, pi:pi+10, :] = proj_func(
                    writer_out, resid_vec
                )

    return projection_matrix

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

node_resid_projections = calc_node_resid_projection(prompts_t, projection)
plot_projection_node_resid(node_resid_projections, projection)



# %%
# Calculate cosine similarity between W_in and W_out

W_in: Float[Tensor, 'layer dmodel dmlp'] = model.W_in
W_out: Float[Tensor, 'layer dmlp dmodel'] = model.W_out

neuron_weight_cosine_similarity = einops.einsum(
    W_in / W_in.norm(dim=-2, keepdim=True),
    W_out / W_out.norm(dim=-1, keepdim=True),
    "layer dmodel dmlp, layer dmlp dmodel -> layer dmlp"
)
# %%
px.histogram(
    neuron_weight_cosine_similarity.flatten().cpu().numpy(), 
    title="Cosine Similarity between W_in and W_out"
).show()
# %%
# neuron_weight_cosine_similarity.flatten()[neuron_weight_cosine_similarity.flatten() < -0.4].sum()
(neuron_weight_cosine_similarity.flatten() < -0.4).sum()
# %%
