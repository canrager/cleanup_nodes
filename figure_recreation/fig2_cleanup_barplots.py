#%%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path
parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import torch
import einops

from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from plotting import ntensor_to_long
from jamesd_utils import projection_ratio

import matplotlib.pyplot as plt
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 240
N_CODE_PROMPTS = 60
WRITER_NAME = "L0H2"
FIG_FILEPATH = "figs/fig2_cleanup_barplots.jpg"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


#%%
def get_node_from_cache(cache, node_name: str) -> torch.Tensor:
    if node_name.startswith("L"):
        layer, head = node_name[1:].split("H")
        layer = int(layer)
        head = int(head)
        return cache[f'blocks.{layer}.attn.hook_result'][:, :, head, :].unsqueeze(0)
    elif node_name.startswith("MLP"):
        layer = int(node_name[3:])
        return cache[f'blocks.{layer}.hook_mlp_out'].unsqueeze(0)


#%%
prompts = get_prompts_t(
    n_text_prompts=N_TEXT_PROMPTS,
    n_code_prompts=N_CODE_PROMPTS,
).to(device)

# Throws a warning if there is a non-unique prompt
if not (torch.unique(prompts, dim=0).shape == prompts.shape):
    print("WARNING: at least 1 prompt is not unique")

#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
model.cfg.use_attn_result = True

#%%
hook_substrs = [
    "attn.hook_result",
    "mlp_out",
]

# Run a forward pass and cache selected activations
_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: any(substr in name for substr in hook_substrs),
    device=device,
)

#%%
# Create a tensor of nodes
nodes_list = []
for l in range(model.cfg.n_layers):
    heads = einops.rearrange(
        cache[f'blocks.{l}.attn.hook_result'],
        "batch pos head d_model -> head batch pos d_model"
    )
    mlp = einops.rearrange(
        cache[f'blocks.{l}.hook_mlp_out'],
        "batch pos d_model -> 1 batch pos d_model"
    )
    nodes_list.extend([heads, mlp])

all_nodes = torch.cat(nodes_list, dim=0)  # (n_nodes, batch, pos, d_model)

# Create node names
node_names = []
for l in range(model.cfg.n_layers):
    for h in range(model.cfg.n_heads):
        node_names.append(f"L{l}H{h}")
    node_names.append(f"MLP{l}")


# Get the writer node's tensor from a given node name
writer_node = get_node_from_cache(cache, WRITER_NAME)

# Calculate reinforcement ratios off all nodes projected onto the writer node
proj_ratios = projection_ratio(all_nodes, writer_node)  # shape: (resid node batch pos)

# Get RRs into a long format df
df = ntensor_to_long(
    proj_ratios,
    value_name="projection_ratio",
    dim_names=["node", "batch", "pos"],
)

layers, comps = divmod(df.node, model.cfg.n_heads + 1)
df["layer"] = layers
df["comp"] = comps
df["node_name"] = df.node.map(lambda x: node_names[x])

# Create node names for x ticks
node_names_without = node_names.copy()
node_names_without.remove(WRITER_NAME)

#%% Not used in final plot, but in the plot description
df_sum_cleaners = (df
    # [df.node_name.isin(["L2H2", "L2H4", "L2H5", "L2H6", "L2H7"])]
    [df.node_name.isin(["L2H2", "L2H3", "L2H4", "L2H5", "L2H6", "L2H7"])]
    # [df.node_name.isin(["L2H0", "L2H1", "L2H2", "L2H3", "L2H4", "L2H5", "L2H6", "L2H7"])]
    [["projection_ratio", "batch", "pos", "node_name"]]
    .groupby(["batch", "pos"]).sum()
    .reset_index()
    [["projection_ratio", "batch", "pos"]]
)
cleaner_median = df_sum_cleaners["projection_ratio"].median()

#%%
# Create figure
fig, ax = plt.subplots(figsize=(12, 5))

sns.barplot(
    data=df.query("node_name != @WRITER_NAME"),
    x="node_name",
    y="projection_ratio",
    estimator="median",
    errorbar=("pi", 75),
    order=node_names_without,
    ax=ax,
)

# Add vlines to separate layers
for l in [node_names_without.index(f"L{i}H0") for i in range(1, model.cfg.n_layers)]:
    ax.axvline(l - 0.5, color="black", linewidth=1, linestyle="--")

# Rotate x-axis labels
for tick in ax.get_xticklabels():
    tick.set_rotation(90)

ax.set_title(
    (
        f"Projection ratios of all nodes (except {WRITER_NAME}) projected onto {WRITER_NAME}"
        # f"Median PR of sum(L2H{{2,3,4,5,6,7}}) = {cleaner_median:.2f}"
    ),
    fontsize=16,
)
ax.set_xlabel("node")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel("PR(node, L0H2)", fontsize=12);

#%%
# Save figure
print(f"Cleaners median: {cleaner_median}")
fig.savefig(FIG_FILEPATH, bbox_inches="tight")
print("Saved figure to file: ", FIG_FILEPATH)
