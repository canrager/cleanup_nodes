#%%
# Gross code to allow for importing from parent directory
import os, sys
from pathlib import Path
parent_path = str(Path(os.getcwd()).parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)

# Imports
import gc
import torch
import einops
import numpy as np

from transformer_lens import HookedTransformer
from load_data import get_prompts_t
from plotting import ntensor_to_long
from jamesd_utils import projection_ratio_cartesian

from plotly.graph_objs.layout._shape import Shape
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# Global settings and variables
sns.set()
torch.set_grad_enabled(False)
device = "cpu"

N_TEXT_PROMPTS = 2
N_CODE_PROMPTS = 1
FIG_FILEPATH = "figs/fig1_resid_onto_node_plotly.html"

# Transformer Lens model names:
# https://github.com/neelnanda-io/TransformerLens/blob/3cd943628b5c415585c8ef100f65989f6adc7f75/transformer_lens/loading_from_pretrained.py#L127
MODEL_NAME = "gelu-4l"


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

# Generate names of residual stream locations
resid_names = ["blocks.0.hook_resid_pre"]
for i in range(model.cfg.n_layers):
    resid_names.append(f"blocks.{i}.hook_resid_mid")
    resid_names.append(f"blocks.{i}.hook_resid_post")

# Generate names of attention blocks for each layer
attn_names = []
for i in range(model.cfg.n_layers):
    attn_names.append(f"blocks.{i}.attn.hook_result")

# Generate names of MLP out for each layer
mlp_names = []
for i in range(model.cfg.n_layers):
    mlp_names.append(f"blocks.{i}.hook_mlp_out")

hook_names = resid_names + attn_names + mlp_names

# Run a forward pass and cache selected activations
_, cache = model.run_with_cache(
    prompts,
    names_filter=lambda name: name in hook_names,
    device=device,
)
del _
gc.collect()

#%%
# Concatenate the cached activations into tensors
cache_resids = torch.stack(
    [cache[name] for name in resid_names],
    dim=0,
)  # shape: (resid, batch, pos, d_model)

# Generate node names by layer
node_names = []
for l in range(model.cfg.n_layers):
    node_names.extend([f"H{l}.{h}" for h in range(model.cfg.n_heads)])
    node_names.append(f"MLP_{l}")

# Generate node tensors by layer, where the node dimension aligns with `node_names`
node_tensors = []
for l in range(model.cfg.n_layers):
    attn_heads_of_layer = cache[f"blocks.{l}.attn.hook_result"]
    attn_heads_of_layer = einops.rearrange(
        attn_heads_of_layer, "batch pos head d_model -> head batch pos d_model",
    )
    mlp_of_layer = cache[f"blocks.{l}.hook_mlp_out"].unsqueeze(0)
    node_tensors.append(torch.cat([attn_heads_of_layer, mlp_of_layer], dim=0))

cache_nodes = torch.cat(node_tensors, dim=0)  # shape: (node, batch, pos, d_model)

# All dims except the 0th dim should be the same
assert cache_nodes.ndim == cache_resids.ndim
assert all([d1 == d2 for d1, d2 in zip(cache_nodes.shape[1:], cache_resids.shape[1:])])

#%%
# Calculate projection ratios
reinf_ratios = projection_ratio_cartesian(cache_resids, cache_nodes)  # shape: (resid node batch pos)
del cache, cache_resids, cache_nodes
gc.collect()

df = ntensor_to_long(
    reinf_ratios,
    value_name="projection_ratio",
    dim_names=["resid", "node", "batch", "pos"],
)

layers, nodes = divmod(df.node, model.cfg.n_heads + 1)
df["layer"] = layers
df["node"] = nodes

#%%
# Calculate quantiles
quantile_fncs = [
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.50),
    lambda x: x.quantile(0.75),
]

df_quantiles = (
    df.groupby(["resid", "layer", "node"])
    .agg({"projection_ratio": quantile_fncs})
)
df_quantiles.columns = ["rr_q25", "rr_q50", "rr_q75"]
df_quantiles = df_quantiles.reset_index()

df_quantiles["error_plus"] = df_quantiles.rr_q75 - df_quantiles.rr_q50
df_quantiles["error_minus"] = df_quantiles.rr_q50 - df_quantiles.rr_q25

# Set node names
df_quantiles["node_name"] = "H" + df_quantiles.layer.astype(str) + "." + df_quantiles.node.astype(str)
df_quantiles["node_name"] = df_quantiles.node_name.where(
    df_quantiles.node != model.cfg.n_heads,
    "MLP" + df_quantiles.layer.astype(str),
)

#%%
# Create main figure
main_fig = go.Figure()

# Add all traces to main figure
for l in range(model.cfg.n_layers):
    tmp_fig = px.line(
        data_frame=df_quantiles.query(f"layer == {l}"),
        x="resid",
        y="rr_q50",
        color="node_name",
        error_y="error_plus",
        error_y_minus="error_minus",
    )

    for trace in tmp_fig.select_traces():
        main_fig.add_trace(trace)

# Set title on start
title = (
    f"Various residual stream locations, projected onto the outputs of "
    f"nodes in layer 0"
)
main_fig.update_layout(title={"text": title}
)
main_fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
main_fig.add_vline(x=1.5, line_dash="dot", line_color="grey")

# Which traces to show on start
layers_to_show = np.zeros(model.cfg.n_layers, dtype=bool)
layers_to_show[0] = True
traces_to_show = np.repeat(layers_to_show, model.cfg.n_heads + 1)

for data, to_show in zip(main_fig.data, traces_to_show):
    data.visible = to_show

# Create and add slider
slider_steps = []
for i in range(model.cfg.n_layers):
    # Which traces to show on slider update
    layers_to_show = np.zeros(model.cfg.n_layers, dtype=bool)
    layers_to_show[i] = True
    traces_to_show = np.repeat(layers_to_show, model.cfg.n_heads + 1)

    title = (
        f"Various residual stream locations, projected onto the outputs of "
        f"nodes in layer {i}"
        f"Averaged across batch (n={prompts.shape[0]}) and position (n={prompts.shape[1]})"
    )

    shapes = []
    for j, style in zip([0, 1], ["dash", "dot"]):
        shapes.append(Shape({
            'line': {'color': 'grey', 'dash': style},
            'type': 'line',
            'x0': 0.5 + i*2 + j,
            'x1': 0.5 + i*2 + j,
            'xref': 'x',
            'y0': 0,
            'y1': 1,
            'yref': 'y domain'
        }))

    step = dict(
        method="update",
        label=f"Layer {i}",
        args=[
            {"visible": traces_to_show},
            {"title": title},
            {"shapes": shapes}
        ],  # Layout attributes
    )

    step["args"][1]["shapes"] = shapes

    slider_steps.append(step)

sliders = [dict(
    active=0,
    steps=slider_steps,
)]

# Prep main figure
middle_ticks = []
for l in range(model.cfg.n_layers):
    middle_ticks.append(f"H{l}.X")
    middle_ticks.append(f"MLP{l}")
xaxis_ticknames = [val for pair in zip(resid_names, middle_ticks) for val in pair] + [resid_names[-1]]

main_fig.update_layout(
    sliders=sliders,
    xaxis_title="Residual Stream Location",
    yaxis_title="Projection Ratio",
    xaxis=dict(
        tickmode="array",
        tickvals=np.arange(0, model.cfg.n_heads + 0.1, 0.5),
        ticktext=xaxis_ticknames,
    ),
)

# Reposition the slider
main_fig["layout"]["sliders"][0]["pad"] = dict(t=120)

#%%
# Write the figure to file
main_fig.write_html(FIG_FILEPATH)
print("Saved figure to file: ", FIG_FILEPATH)
