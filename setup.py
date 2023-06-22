# %%
import torch
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformer_lens import HookedTransformer
from pathlib import Path
import pickle
import plotly.express as px
from tqdm.notebook import trange, tqdm
from functools import partial
import pandas as pd
import re
import unicodedata
from functools import partial
import random

model = HookedTransformer.from_pretrained('gelu-3l')
# %%
from datasets import load_dataset

def get_prompts_list(dataset_name: str, n_prompts: int, shuffle_buffer_size: int, shuffle_seed: int):
    print(f"Loading {n_prompts} prompts from {dataset_name}...")
    file_name = f"{dataset_name}-{n_prompts}-seed{shuffle_seed}-buffer{shuffle_buffer_size}.pkl"
    file_path = "./data" / Path(file_name) # Change based on user
    if file_path.exists():
        print("Using pickled prompts...")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    print("Downloading from HuggingFace...")
    prompts_list = []
    ds_unsfuffled = load_dataset(f"NeelNanda/{dataset_name}", streaming=True, split="train")
    ds = ds_unsfuffled.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    ds_iter = iter(ds)
    for _ in trange(n_prompts):
        prompts_list.append(next(ds_iter)["tokens"])
    with open(file_path, "wb") as f:
        pickle.dump(prompts_list, f)
    return prompts_list
# %%
N_TOTAL_PROMPTS = 100
N_C4_TOTAL_PROMPTS = int(0.8 * N_TOTAL_PROMPTS)
N_CODE_TOTAL_PROMPTS = N_TOTAL_PROMPTS - N_C4_TOTAL_PROMPTS
DS_SHUFFLE_SEED, DS_SHUFFLE_BUFFER_SIZE = 5235, N_TOTAL_PROMPTS // 10

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

def get_token_counts(prompts_t_):
    unique_tokens, tokens_counts_ = torch.unique(prompts_t_, return_counts=True)
    tokens_counts = torch.zeros(model.cfg.d_vocab, dtype=torch.int64, device=device)
    tokens_counts[unique_tokens] = tokens_counts_.to(device)
    return tokens_counts

prompts_t = get_prompts_t()
token_counts = get_token_counts(prompts_t)

MIN_TOKEN_COUNT = N_TOTAL_PROMPTS // 1_000
tokens = torch.arange(model.cfg.d_vocab, device=device, dtype=torch.int32)
tokens = tokens[token_counts >= MIN_TOKEN_COUNT]
tokens_set = set(tokens.tolist())
prompts_t[0, 1]
# %%
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