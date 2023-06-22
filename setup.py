# %% Import modules
import torch
import einops
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformer_lens import HookedTransformer, ActivationCache
import plotly.express as px
from tqdm.notebook import trange, tqdm
from datasets import load_dataset

from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import List, Callable
from plotting import get_fig_head_mlp_neuron


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

# %%
# is outdated as we can run_with_cache and apply filter
def get_raw_patterns(model, prompts, layer: int, head: int):
    patterns = None
    def hook_get_pattern(act, hook):
        # batch, dst, src
        nonlocal patterns
        patterns = act[:, head]

    model.reset_hooks()
    model.add_hook(f"blocks.{layer}.attn.hook_pattern", hook_get_pattern)
    model(prompts, stop_at_layer=layer+1)
    model.reset_hooks()

    return patterns


def get_patterns(model, 
                prompts, # n_examples, n_ctx
                layer: int, 
                head: int, 
                mb_size: int):

    # Store attention patterns as a list of dicts
    patterns = []
    
    # Enumerate over each batch of prompts
    for prompts_mb_idx in trange(0, prompts.shape[0], mb_size):
        prompts_mb = prompts[prompts_mb_idx:prompts_mb_idx+mb_size]

        # Get attn patterns of a specific head, ignoring first ignore_first_n_pos dst pos or rows
        raw_patterns_mb = get_raw_patterns(model, prompts_mb, layer, head)
        # TODO: ...


#%% Inpect dataset

# prompts_t[0]
# print("".join(model.to_str_tokens(prompts_t[10])))

# %% Model inspection

logits, activation_cache = model.run_with_cache(prompts_t[10])
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

# %% Get pattern with transformerlens
def get_node_outputs_and_resid(name: str) -> bool:
    hook_names = ["result", "post", "resid_mid", "resid_post"]
    return any(hook_name in name for hook_name in hook_names)

logits, activation_cache = model.run_with_cache(
    prompts_t[:10],
    names_filter= get_node_outputs_and_resid)

# %%
activation_cache.keys()


# %% Projection functions
# %% Node - node projection
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

# prompt 0, 10th position
## all attentions
# attn_hook_names = [name for name in activation_cache.keys() if "result" in name]
# attn_activations = [activation_cache[name] for name in attn_hook_names]
# attn_activations = einops.rearrange(attn_activations, "layer batch pos head dmodel -> batch pos (layer head) dmodel")

# mlp_hook_names = [name for name in activation_cache.keys() if "mlp_out" in name]
# mlp_activations = [activation_cache[name] for name in mlp_hook_names]
# mlp_activations = einops.rearrange(mlp_activations, "layer batch pos head dmodel -> batch pos (layer head) dmodel")


# for attn_layer

# batch, pos, writer, cleaner


# %%
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
# %%

node_node_projections = calc_node_node_projection(prompts_t, proj_func=projection)

#%%
get_fig_head_mlp_neuron(
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
