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
from datasets import load_dataset

from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Callable
from plotting import get_fig_head_to_mlp_neuron

#%% Setup model
model = HookedTransformer.from_pretrained('gelu-4l')
model.cfg.use_attn_result = True

# %% Setup dataset
def get_prompts_list(dataset_name: str, n_prompts: int, shuffle_buffer_size: int, shuffle_seed: int):
    print(f"Loading {n_prompts} prompts from {dataset_name}...")
    # file_name = f"{dataset_name}-{n_prompts}-seed{shuffle_seed}-buffer{shuffle_buffer_size}.pkl"
    # file_path = "./data" / Path(file_name) # Change based on user
    # if file_path.exists():
    #     print("Using pickled prompts...")
    #     with open(file_path, "rb") as f:
    #         return pickle.load(f)
    # print("Downloading from HuggingFace...")
    prompts_list = []
    ds_unshuffled = load_dataset(f"NeelNanda/{dataset_name}", streaming=True, split="train")
    ds = ds_unshuffled.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    ds_iter = iter(ds)
    for _ in trange(n_prompts):
        prompts_list.append(next(ds_iter)["tokens"])
    # with open(file_path, "wb") as f:
    #     pickle.dump(prompts_list, f)
    return prompts_list

# %% Dataset preprocessing
N_TOTAL_PROMPTS = 100
N_C4_TOTAL_PROMPTS = int(0.8 * N_TOTAL_PROMPTS)
N_CODE_TOTAL_PROMPTS = N_TOTAL_PROMPTS - N_C4_TOTAL_PROMPTS
DS_SHUFFLE_SEED, DS_SHUFFLE_BUFFER_SIZE = 5235, N_TOTAL_PROMPTS // 10 # Ds_shuffle_biffersize determines subset of ds, where prompts are ramdomly sampled from

def shuffle_tensor(tensor, dim):
    torch.manual_seed(DS_SHUFFLE_SEED)
    torch.cuda.manual_seed(DS_SHUFFLE_SEED)
    return tensor[torch.randperm(tensor.shape[dim])]

def get_prompts_t():
    shuffle_kwargs = dict(shuffle_buffer_size=DS_SHUFFLE_BUFFER_SIZE, shuffle_seed=DS_SHUFFLE_SEED)
    c4_prompts_list = get_prompts_list("c4-tokenized-2b", n_prompts=N_C4_TOTAL_PROMPTS, **shuffle_kwargs)
    code_prompts_list = get_prompts_list("code-tokenized", n_prompts=N_CODE_TOTAL_PROMPTS, **shuffle_kwargs)
    prompts_t = torch.tensor(
        c4_prompts_list + code_prompts_list
    )
    return shuffle_tensor(prompts_t, dim=0)

def get_token_counts(prompts_t_): # returns list of #occurences per token
    unique_tokens, tokens_counts_ = torch.unique(prompts_t_, return_counts=True)
    tokens_counts = torch.zeros(model.cfg.d_vocab, dtype=torch.int64, device=device)
    tokens_counts[unique_tokens] = tokens_counts_.to(device)
    return tokens_counts

prompts_t = get_prompts_t()
token_counts = get_token_counts(prompts_t)

# filter out tokens that occur less than 0.1% than the total number of prompts
MIN_TOKEN_COUNT = N_TOTAL_PROMPTS // 1_000
tokens = torch.arange(model.cfg.d_vocab, device=device, dtype=torch.int32)
tokens = tokens[token_counts >= MIN_TOKEN_COUNT]
tokens_set = set(tokens.tolist())
prompts_t[0, 1]


#%% Inpect dataset

# prompts_t[0]
# print("".join(model.to_str_tokens(prompts_t[10])))

# %% Model inspection

# logits, activation_cache = model.run_with_cache(prompts_t[10])
# print(logits.shape)

# print("".join(model.to_str_tokens(prompts_t[10])))
# print("prediction:", model.to_str_tokens(logits.argmax(dim=-1)[0, -1]))

# %% Implement get_neuron_output function, get neuron output for a specific neuron

def get_neuron_output(
    cache: ActivationCache, layer_idx: int, neuron_idx: int
) -> Float[Tensor, "batch pos dmodel"]:
    neuron_activation = cache["mlp_post", layer_idx][:, :, neuron_idx]
    neuron_wout = model.W_out[layer_idx, neuron_idx]
    mlp_bias = model.b_out[layer_idx] / model.cfg.d_mlp         # TODO design choice, discuss with Jett

    # print(f"{neuron_activation.shape=}")
    # print(f"{neuron_wout.shape=}")
    # print(f"{mlp_bias.shape=}")

    neuron_out = neuron_activation.unsqueeze(dim=-1) * neuron_wout + mlp_bias
    # print(neuron_activation.shape)
    return neuron_out


# test our get_neuron_output function
# custom_mlp_out = sum(
#     [get_neuron_output(activation_cache, layer, neuron_idx) for neuron_idx in range(model.cfg.d_mlp)]
# )

# torch.isclose(custom_mlp_out, activation_cache["mlp_out", layer], atol= 1e-5).all()

# %% Get activation cache with transformerlens
def get_node_outputs_and_resid(name: str) -> bool:
    hook_names = ["result", "post", "resid_mid", "resid_post"]
    return any(hook_name in name for hook_name in hook_names)

logits, activation_cache = model.run_with_cache(
    prompts_t[:10],
    names_filter= get_node_outputs_and_resid)

activation_cache.keys()

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

# %% Node - Node Projections
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
            tokens[pi : pi + 10], names_filter=get_node_outputs_and_resid
        )

        for w, (lw, hw) in enumerate(all_heads):
            for c, (lc, hc) in enumerate(all_neurons):
                writer_out = cache['result', lw][:, :, hw, :]
                cleaner_out = get_neuron_output(cache, lc, hc)
                projection_matrix[w, c, pi:pi+10, :] = proj_func(
                    writer_out, cleaner_out
                )

    return projection_matrix
# %% Calculate node-node projections data
node_node_projections = calc_node_node_projection(prompts_t, proj_func=projection)

#%% Plot node-node projections
get_fig_head_to_mlp_neuron(
    projections=node_node_projections,
    quantile=0.1,
    k=50,
    n_heads=model.cfg.n_heads,
    d_mlp=model.cfg.d_mlp
).show()

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
            tokens[pi : pi + 10], names_filter=get_node_outputs_and_resid
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

def plot_projection(proj_func: Callable):
    node_resid_projection_matrix = calc_node_resid_projection(prompts_t, proj_func)
    px.imshow(
        node_resid_projection_matrix.flatten(start_dim=-2).mean(dim=-1),
        color_continuous_midpoint=0,
        color_continuous_scale='RdBu',
        y=[f"Head {layer}.{head}" for layer, head in all_heads],
        x=[f"Resid {layer}.{resid.split('_')[-1]}" for layer, resid in all_resids],
        title=f"Projection of Resid to Attn Head Output Direction using {proj_func.__name__}"
    ).show()

plot_projection(projection)
plot_projection(cos_similarity)



# %%
