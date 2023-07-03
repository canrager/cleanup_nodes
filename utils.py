from transformer_lens import HookedTransformer, ActivationCache
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Callable
import torch
import einops
from tqdm.notebook import trange
import plotly.express as px

# %% Implement get_neuron_output function, get neuron output for a specific neuron


def get_neuron_output(
    cache: ActivationCache, model: HookedTransformer, layer_idx: int, neuron_idx: int
) -> Float[Tensor, "batch pos dmodel"]:
    neuron_activation = cache["mlp_post", layer_idx][:, :, neuron_idx]
    neuron_wout = model.W_out[layer_idx, neuron_idx]
    mlp_bias = (
        model.b_out[layer_idx] / model.cfg.d_mlp
    )  # TODO design choice, discuss with Jett

    neuron_out = neuron_activation.unsqueeze(dim=-1) * neuron_wout + mlp_bias
    return neuron_out

# %% Get pattern with transformerlens
def act_filter(name: str) -> bool:
    hook_names = ["result", "post", "resid_mid", "resid_post"]
    return any(hook_name in name for hook_name in hook_names)


# %% Projection functions
def projection(
    writer_out: Float[Tensor, "batch pos dmodel"],
    cleaner_out: Float[Tensor, "batch pos dmodel"],
) -> Float[Tensor, "batch pos"]:
    """Compute the projection from the cleanup output vector to the writer output direction"""
    norm_writer_out = torch.norm(writer_out, dim=-1, keepdim=True)
    dot_prod = einops.einsum(
        writer_out / norm_writer_out,
        cleaner_out,
        "batch pos dmodel, batch pos dmodel -> batch pos",
    )
    return dot_prod

# from james_util
def projection_ratio(
    cleaner_vectors: Float[Tensor, "... d_model"],
    writer_vectors: Float[Tensor, "... d_model"],
) -> Float[Tensor, "..."]:
    """
    Works element-wise on the last dimension of the input tensors.

    Interpretation: 
    After you project `cleaner_vectors` onto the direction of
    `writer_vectors`, the projection ratio is the ratio of length of the
    projection to the length of `writer_vectors`.

    A projection ratio of -1 means that the projection is in the opposite
    direction of `writer_vectors`, and has the same L2 norm.
    
    Mathematically:
    projection_ratio(A, B) = (A⋅B) / ||B||^2
        OR
    projection_ratio(A, B) = cos_sim(A, B) * ||A|| / ||B||


    Args:
        cleaner_vectors: a tensor of vectors that will be projected
        writer_vectors: a tensor of vectors that will be projected onto
    """
    return einops.einsum(
        cleaner_vectors,
        writer_vectors / writer_vectors.norm(dim=-1, keepdim=True).pow(2),
        "... d_model, ... d_model -> ...",
    )

def reinforcement_ratio(
    writer_out: Float[Tensor, "batch pos dmodel"],
    cleaner_out: Float[Tensor, "batch pos dmodel"],
):
    norm_writer_out = torch.norm(writer_out, dim=-1, keepdim=True)
    dot_prod = einops.einsum(
        writer_out / norm_writer_out.pow(2),
        cleaner_out,
        "batch pos dmodel, batch pos dmodel -> batch pos",
    )
    return dot_prod


def cos_similarity(
    writer_out: Float[Tensor, "batch pos dmodel"],
    cleanup_out: Float[Tensor, "batch pos dmodel"],
) -> Float[Tensor, "batch pos"]:
    """Compute the projection from the cleanup output vector to the writer output direction"""
    norm_writer_out = torch.norm(writer_out, dim=-1, keepdim=True)
    norm_cleaner_out = torch.norm(cleanup_out, dim=-1, keepdim=True)
    dot_prod = einops.einsum(
        writer_out / norm_writer_out,
        cleanup_out / norm_cleaner_out,
        "batch pos dmodel, batch pos dmodel -> batch pos",
    )
    return dot_prod


# %%
# Calculate cosine similarity between W_in and W_out


def select_neurons_by_cosine_similarity(
    model: HookedTransformer, thres: float = -0.4, show_distribution: bool = False
) -> List[tuple]:
    W_in: Float[Tensor, "layer dmodel dmlp"] = model.W_in
    W_out: Float[Tensor, "layer dmlp dmodel"] = model.W_out

    neuron_weight_cosine_similarity = einops.einsum(
        W_in / W_in.norm(dim=-2, keepdim=True),
        W_out / W_out.norm(dim=-1, keepdim=True),
        "layer dmodel dmlp, layer dmlp dmodel -> layer dmlp",
    )

    # show distribution of neuron cosine similarity
    if show_distribution:
        px.histogram(
            neuron_weight_cosine_similarity.flatten().cpu().numpy(),
            title="Cosine Similarity between W_in and W_out",
        ).show()

    # select neurons with large negative cosine similarity
    print(
        f"Number of neurons with cosine similarity less than {thres} for each layer:\n",
        (neuron_weight_cosine_similarity < thres).sum(dim=-1),
    )

    selected_neurons = [
        (l, n)
        for l in range(model.cfg.n_layers)
        for n in range(model.cfg.d_mlp)
        if neuron_weight_cosine_similarity[l, n] < thres
    ]

    return selected_neurons


# %% Calculate node-node projection matrix
def calc_node_node_projection(
    model: HookedTransformer,
    tokens: List[int],
    selected_neurons: List[tuple],
    proj_func: Callable,
) -> Float[Tensor, "head neuron prompt pos"]:
    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]  # [head0.0, head0.1, ..., head2.7]

    n_total_heads = len(all_heads)
    n_selected_neurons = len(selected_neurons)
    n_prompts = len(tokens)
    n_pos = len(tokens[0])

    projection_matrix = torch.zeros(
        n_total_heads,
        n_selected_neurons,
        n_prompts,
        n_pos,
    )

    for pi in trange(0, len(tokens), 10):
        _, cache = model.run_with_cache(tokens[pi : pi + 10], names_filter=act_filter)

        for w, (lw, hw) in enumerate(all_heads):
            for c, (lc, hc) in enumerate(selected_neurons):
                writer_out = cache["result", lw][:, :, hw, :]
                cleaner_out = get_neuron_output(cache, model, lc, hc)
                projection_matrix[w, c, pi : pi + 10, :] = proj_func(
                    writer_out, cleaner_out
                )

    return projection_matrix


def calc_node_resid_projection(
    model: HookedTransformer,
    tokens: List[int],
    proj_func: Callable,
) -> Float[Tensor, "head resid prompt pos"]:
    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]  # [head0.0, head0.1, ..., head2.7]

    all_resids = [
        (l, act)
        for l in range(model.cfg.n_layers)
        for act in ["resid_mid", "resid_post"]
    ]  # [resid_mid0, resid_post0, resid_mid1, ... ,resid_post2]

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
        _, cache = model.run_with_cache(tokens[pi : pi + 10], names_filter=act_filter)

        for w, (lw, hw) in enumerate(all_heads):
            for resid_idx, (resid_layer, resid_name) in enumerate(all_resids):
                writer_out = cache["result", lw][:, :, hw, :]
                resid_vec = cache[resid_name, resid_layer]
                projection_matrix[w, resid_idx, pi : pi + 10, :] = proj_func(
                    writer_out, resid_vec
                )

    return projection_matrix


# %% Plot node-node projection matrix
# TODO: move to plotting.py


def plot_projection_node_resid(
    model: HookedTransformer,
    node_projections: Float[Tensor, "head resid prompt pos"],
    proj_func: Callable,
):
    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]  # [head0.0, head0.1, ..., head2.7]

    all_resids = [
        (l, act)
        for l in range(model.cfg.n_layers)
        for act in ["resid_mid", "resid_post"]
    ]  # [resid_mid0, resid_post0, resid_mid1, ... ,resid_post2]

    px.imshow(
        node_projections.flatten(start_dim=-2).mean(dim=-1),
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        y=[f"Head {layer}.{head}" for layer, head in all_heads],
        x=[f"Resid {layer}.{resid.split('_')[-1]}" for layer, resid in all_resids],
        title=f"Projection of Resid to Attn Head Output Direction using {proj_func.__name__}",
    ).show()
