"""
TODO:
- Want a plot the tells us which layer has neurons doing cleanup
  - Fix heads, fix neurons but select top N neurons by various metrics:
     - .max(dim=head)
- Try different neuron ranking metrics for `get_fig_head_to_mlp_neuron`
  - Currently using quantile
  - Can try proportion of (batch*pos) greater/lesser than some threshold

"""

# %%
import itertools
from typing import Tuple
import numpy as np
import torch as t
import pandas as pd

import einops
from jaxtyping import Float
from typing import Optional, Union
from torch import Tensor

import plotly.express as px
import plotly.graph_objects as go

from transformer_lens.utils import get_act_name
from utils import projection


def _neuron_index_to_name(neuron_index: int, d_mlp: int) -> str:
    layer, neuron = divmod(neuron_index, d_mlp)
    return f"N{layer}.{neuron}"


def _head_index_to_name(head_index: int, n_heads: int) -> str:
    layer, head = divmod(head_index, n_heads)
    return f"H{layer}.{head}"


def get_fig_head_to_mlp_neuron(
    projections: Float[t.Tensor, "head neuron prompt pos"],
    quantile: float,
    k: int,
    n_heads: int,
    d_mlp: int,
) -> go.Figure:
    """
    Plot the top k neurons for each head, where the top k neurons are
    determined by the quantile of the projection matrix.

    Args:
        projections: a tensor of projections
        quantile: the quantile across prompts * positions of each neuron
            `0.01` means bottom 1%, `0.99` means top 1%
        k: the number of neurons to plot for each head
        n_heads: the number of heads per layer
        d_mlp: the number of neurons in the MLPs intermediate layer

    Returns:
        A plotly figure
    """

    # Get quantiles
    projections_quantile = einops.rearrange(
        projections,
        "head neuron prompt pos -> head neuron (prompt pos)",
    ).quantile(quantile, dim=-1)

    # Get top k neurons for each head
    #
    projections_topk = projections_quantile.topk(k, dim=-1, largest=quantile >= 0.5)

    # Create head names and neuron names for hover text
    nrows, ncols = projections_topk.values.shape
    head_names = [_head_index_to_name(i, n_heads) for i in range(projections.shape[0])]

    neuron_names = np.empty(projections_topk.indices.shape, dtype="<U10")
    for row, col in itertools.product(range(nrows), range(ncols)):
        neuron_index = projections_topk.indices[row, col].item()
        neuron_names[row, col] = _neuron_index_to_name(neuron_index, d_mlp)

    # Compute some plotting params
    absmax = projections_topk.values.abs().max().item()

    fig = px.imshow(
        projections_topk.values,
        y=head_names,
        labels=dict(x=f"Neuron Rank", y="Head"),
        zmin=-absmax,
        zmax=absmax,
        color_continuous_scale="RdBu",
        title=f"Top {k} Quantiles of Neurons (Head to Neuron, q={quantile})",
    )

    # Add hover text
    hovertemplate = (
        "Metric Name: %{z}<br><br>"
        "Head Name: %{y}<br>"
        "Neuron Name: %{customdata}<br>"
        "Neuron Rank: %{x}<br>"
    )
    data = [{"hovertemplate": hovertemplate, "customdata": neuron_names}]
    fig.update(data=data)

    return fig


def get_fig_head_to_mlp_neuron_by_layer(
    projections: Float[t.Tensor, "head neuron prompt pos"],
    k: int,
    quantile: float,
    n_layers: int,
) -> go.Figure:
    """
    Plot the top k neurons for each head grouped by later, where the top k
    neurons are determined by the quantile of the projection matrix.

    Args:
        projections: a tensor of projections
        k: the number of neurons to plot for each head
        quantile: the quantile across prompts * positions of each neuron
            `0.01` means bottom 1%, `0.99` means top 1%
        n_layers: the number of layers

    Returns:
        A plotly figure
    """
    projections_quantile = einops.rearrange(
        projections,
        "head (layer n) batch pos -> head layer n (batch pos)",
        layer=n_layers,
    ).quantile(
        quantile, dim=-1
    )  # shape: (head, layer, neuron_in_layer)

    largest_first = quantile >= 0.5
    projections_topk = projections_quantile.topk(k, dim=-1, largest=largest_first)

    # Create head names for hover text
    n_heads = projections.shape[0] // n_layers
    head_names = [_head_index_to_name(i, n_heads) for i in range(projections.shape[0])]

    # Create neuron names for hover text
    neuron_names = []
    heads, layers, neurons = projections_topk.indices.shape
    for head in range(heads):
        row = []
        for layer, neuron in itertools.product(range(layers), range(neurons)):
            neuron_index = projections_topk.indices[head, layer, neuron].item()
            name = f"N{layer}.{neuron_index}"
            row.append(name)
        neuron_names.append(row)

    # Reshape for 2D heatmap plotting
    projections_topk_values = einops.rearrange(
        projections_topk.values, "head layer n -> head (layer n)"
    )

    # Create main figure
    absmax = projections_topk_values.abs().max().item()
    title = f"Top {k} Neurons per Layer<br>" f"Batch-pos aggregation: q={quantile}"

    fig = px.imshow(
        projections_topk_values,
        y=head_names,
        labels=dict(x=f"Layer", y="Head"),
        zmin=-absmax,
        zmax=absmax,
        color_continuous_scale="RdBu",
        title=title,
    )

    # Add hover text
    hovertemplate = (
        "Projection Value: %{z}<br><br>"
        "Head Name: %{y}<br>"
        "Neuron Name: %{customdata}<br>"
    )
    data = [{"hovertemplate": hovertemplate, "customdata": neuron_names}]
    fig.update(data=data)

    # Hide x ticks
    fig.update_layout({"xaxis": {"showticklabels": False}})

    # Add vertical and horizontal lines to separate layers
    vline_positions = [x for x in range(k, k * n_layers, k)]
    for vpos in vline_positions:
        fig.add_vline(x=vpos - 0.5, line_dash="dash", line_color="black")

    hline_positions = [x for x in range(n_heads, n_heads * n_layers, n_heads)]
    for hpos in hline_positions:
        fig.add_hline(y=hpos - 0.5, line_dash="dash", line_color="black")

    return fig


def get_fig_head_to_selected_mlp_neuron(
    projections: Float[t.Tensor, "head neuron prompt pos"],
    k: int,
    quantile: float,
    neuron_names: Tuple[int, int],
    n_layers: int,
) -> go.Figure:
    """
    TODO: describe function

    Args:
        projections: a tensor of projection values
        k: the number of neurons to plot for each head
        neuron_names: a tuple of (layer, neuron) indices
        n_layers: the number of layers

    Returns:
        A plotly figure
    """
    # Aggregate along the batch*pos dimenions
    projections_flattened = einops.rearrange(
        projections,
        "head neuron batch pos -> head neuron (batch pos)",
    )
    projections_aggregated = projections_flattened.quantile(quantile, dim=-1)

    # Get top k neurons for each head
    largest_first = quantile >= 0.5
    projections_topk = projections_aggregated.topk(
        k,
        dim=-1,
        largest=largest_first,
        sorted=True,
    )

    # Create head names for hover text
    n_heads = projections.shape[0] // n_layers
    head_names = [_head_index_to_name(i, n_heads) for i in range(projections.shape[0])]

    # Create neuron names for hover text
    neuron_names_string = []
    heads, neurons = projections_topk.indices.shape
    for head in range(heads):
        row = []
        for neuron in range(neurons):
            selected_neuron_index = projections_topk.indices[head, neuron].item()
            layer, neuron_index = neuron_names[selected_neuron_index]
            row.append(f"N{layer}.{neuron_index}")
        neuron_names_string.append(row)

    # Compute some plotting params
    absmax = projections_topk.values.abs().max().item()

    fig = px.imshow(
        projections_topk.values,
        y=head_names,
        labels=dict(x=f"Neuron Rank", y="Head"),
        zmin=-absmax,
        zmax=absmax,
        color_continuous_scale="RdBu",
        title=f"Top {k} Quantiles of Selected Neurons (Head to Neuron, q={quantile})",
    )

    # Add hover text
    hovertemplate = (
        "Metric Name: %{z}<br><br>"
        "Head Name: %{y}<br>"
        "Neuron Name: %{customdata}<br>"
        "Neuron Rank: %{x}<br>"
    )
    data = [{"hovertemplate": hovertemplate, "customdata": neuron_names_string}]
    fig.update(data=data)

    return fig


def single_head_full_resid_projection(
    model,
    prompts,
    writer_layer: int,
    writer_idx: int,
    neuron_layer: Optional[int] = None,
    neuron_idx: Optional[int] = None,
    return_fig: bool = False,
    box_plot: bool = True,
) -> Float[Tensor, "projection_values"]:
    """
    
    """
    # Get act names
    writer_hook_name = get_act_name("result", writer_layer)
    resid_hook_names = ["resid_mid", "resid_post"]

    # Run with cache on all prompts (at once?)
    _, cache = model.run_with_cache(
        prompts,
        names_filter=lambda name: ("resid" in name) or (name == writer_hook_name),
    )

    # calc projections vectorized (is the projection function valid for this vectorized operation?)
    # [ n_resid = 2*n_layers, batch, pos ]

    projections = t.zeros(
        size=(2 * model.cfg.n_layers + 1, len(prompts), model.cfg.n_ctx)
    )

    # add resid pre layer 0
    resid_hook_name = get_act_name("resid_pre", 0)
    projections[0] = projection(
        writer_out=cache[writer_hook_name][:, :, writer_idx, :],
        cleanup_out=cache[resid_hook_name],
    )

    # add resid mid and post for all layers
    for layer in range(model.cfg.n_layers):
        for i, resid_stage in enumerate(resid_hook_names):
            resid_hook_name = get_act_name(resid_stage, layer)

            projections[2 * layer + i + 1] = projection(
                writer_out=cache[writer_hook_name][:, :, writer_idx, :],
                cleanup_out=cache[resid_hook_name],
            )
    projections_full = projections.flatten()  # shape: [n_resid*batch*pos]
    projections = einops.reduce(projections, "n_resid batch pos -> n_resid", "mean")

    if return_fig:
        resid_labels = ["L0_resid_pre"]
        for i in range(model.cfg.n_layers):
            resid_labels.append(f"L{i}_resid_mid")
            resid_labels.append(f"L{i}_resid_post")



        # Set title
        title = f"H{writer_layer}.{writer_idx} projection onto residual stream"
        if neuron_layer and neuron_idx:
            title += f"(linked with N{neuron_layer}.{neuron_idx})"

        if box_plot:
            repeated_labels = np.repeat(resid_labels, len(prompts) * model.cfg.n_ctx)
            df = pd.DataFrame(
                {
                    "projection_value": projections_full.cpu().numpy(),
                    "labels": repeated_labels,
                }
            )
            fig = px.box(
                df,
                x="labels",
                y="projection_value",
                title=title,
            )
        else: # line plot
            d = {"projection_value": projections.cpu().numpy(), "labels": resid_labels}
            df = pd.DataFrame(d)
            fig = px.line(
                df,
                x="labels",
                y="projection_value",
                title=title,
            )
        if neuron_layer and neuron_idx:
            fig.add_vline(x=neuron_layer * 2 + 1, line_dash="dash", line_color="black")

            
        return projections, fig
    else:
        return projections


def ntensor_to_long(tensor: Union[Tensor, np.array]) -> pd.DataFrame:
    """
    Converts an n-dimensional tensor to a long format dataframe.
    Dimension numbering in long format from outer to inner dimension

    Example:
    tensor.shape = [batch pos dmodel]
    df = ntensor_to_long(tensor)
    df.columns
    >>> [values, dim0, dim1, dim2]
    where dims are corresponding to:
    dim0: batch
    dim1: pos
    dim2: dmodel

    """
    df = pd.DataFrame()
    df["values"] = tensor.cpu().numpy().flatten()

    for i, _ in enumerate(tensor.shape):
        pattern = np.repeat(np.arange(tensor.shape[i]), np.prod(tensor.shape[i+1:]))
        n_repeats = np.prod(tensor.shape[:i])
        df[f"dim{i}"] = np.tile(pattern, n_repeats)
    
    return df


# ----- DEMONSTRATION ------------------------------------------------------- #
# %%
if __name__ == "__main__":
    quantile = 0.1
    k = 50

    n_heads = 4
    n_layers = 3
    d_mlp = 32 * 4

    projections = t.randn(n_layers * n_heads, n_layers * d_mlp, 10, 100)

    # fig = get_fig_head_to_mlp_neuron_by_layer(projections, k, quantile, n_layers)
    # fig.show()

    # Generate random neuron_names
    np.random.seed(42)
    neuron_names = []
    for layer in range(n_layers):
        neuron_indices = list(range(d_mlp * 8))
        np.random.shuffle(neuron_indices)
        k_selected = np.random.randint(-50, 50) + projections.shape[1] // n_layers
        k_selected = max(k_selected, 30)
        neuron_indices = sorted(neuron_indices[:k_selected])
        for n in neuron_indices:
            neuron_names.append((layer, n))
    neuron_names = neuron_names[: projections.shape[1]]

    fig = get_fig_head_to_selected_mlp_neuron(projections, k, neuron_names, n_layers)
    fig.show()

def line_with_river(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig