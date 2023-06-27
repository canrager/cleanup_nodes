# %% Setup dataset
import torch

from datasets import load_dataset
from tqdm.auto import trange

from jaxtyping import Float


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

def get_prompts_t(
    n_text_prompts: int = N_C4_TOTAL_PROMPTS,
    n_code_prompts: int = N_CODE_TOTAL_PROMPTS,
    shuffle_buffer_size: int = DS_SHUFFLE_BUFFER_SIZE,
    shuffle_seed: int = DS_SHUFFLE_SEED,
) -> Float[torch.Tensor, "batch pos"]:
    shuffle_kwargs = dict(shuffle_buffer_size=shuffle_buffer_size, shuffle_seed=shuffle_seed)
    c4_prompts_list = get_prompts_list("c4-tokenized-2b", n_prompts=n_text_prompts, **shuffle_kwargs)
    code_prompts_list = get_prompts_list("code-tokenized", n_prompts=n_code_prompts, **shuffle_kwargs)
    prompts_t = torch.tensor(
        c4_prompts_list + code_prompts_list
    )
    return shuffle_tensor(prompts_t, dim=0)

def get_token_counts(model, prompts_t_): # returns list of #occurences per token
    unique_tokens, tokens_counts_ = torch.unique(prompts_t_, return_counts=True)
    tokens_counts = torch.zeros(model.cfg.d_vocab, dtype=torch.int64, device=model.cfg.device)
    tokens_counts[unique_tokens] = tokens_counts_.to(model.cfg.device)
    return tokens_counts


# prompts_t = get_prompts_t()
# token_counts = get_token_counts(model, prompts_t)

# filter out tokens that occur less than 0.1% than the total number of prompts
# MIN_TOKEN_COUNT = N_TOTAL_PROMPTS // 1_000
# tokens = torch.arange(model.cfg.d_vocab, device=device, dtype=torch.int32)
# tokens = tokens[token_counts >= MIN_TOKEN_COUNT]
# tokens_set = set(tokens.tolist())
# prompts_t[0, 1]
