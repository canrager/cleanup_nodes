"""
TODO:
- Want a plot the tells us which layer has neurons doing cleanup
  - Fix heads, fix neurons but select top N neurons by various metrics:
     - .max(dim=head)
- Try different neuron ranking metrics for `get_fig_head_to_mlp_neuron`
  - Currently using quantile
  - Can try proportion of (batch*pos) greater/lesser than some threshold


"""

#%%
import itertools
import numpy as np
import torch as t

import einops
from jaxtyping import Float

import plotly.express as px
import plotly.graph_objects as go


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
    ).quantile(quantile, dim=-1)  # shape: (head, layer, neuron_in_layer)

    largest_first = quantile >= 0.5

    projections_topk = projections_quantile.topk(k, dim=-1, largest=largest_first)

    # Create head names for hover text
    n_heads = projections.shape[0] // n_layers
    head_names = [_head_index_to_name(i, n_heads) for i in range(projections.shape[0])]

    # Create neuron names for hever text
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
    title = (
        f"Top {k} Neurons per Layer<br>"
        f"Batch-pos aggregation: q={quantile}"
    )

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
    fig.update_layout({'xaxis': {'showticklabels': False}})

    # Add vertical and horizontal lines to separate layers
    vline_positions = [x for x in range(k, k * n_layers, k)]
    for vpos in vline_positions:
        fig.add_vline(x=vpos-0.5, line_dash="dash", line_color="black")

    hline_positions = [x for x in range(n_heads, n_heads * n_layers, n_heads)]
    for hpos in hline_positions:
        fig.add_hline(y=hpos-0.5, line_dash="dash", line_color="black")

    return fig




# ----- DEMONSTRATION ------------------------------------------------------- #
#%%
if __name__ == "__main__":

    quantile = 0.1
    k = 10

    n_heads = 4
    n_layers = 3
    d_mlp = 32 * 4

    projections = t.randn(n_layers * n_heads, n_layers * d_mlp, 10, 100)
    
    fig = get_fig_head_to_mlp_neuron_by_layer(projections, k, quantile, n_layers)
    fig.show()
