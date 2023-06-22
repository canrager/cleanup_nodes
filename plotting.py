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
    quantile: int,
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
    head_names = [
        _head_index_to_name(i, n_heads) for i in range(projections_topk.values.shape[0])
    ]
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


# ----- DEMONSTRATION ------------------------------------------------------- #
#%%
if __name__ == "__main__":

    quantile = 0.1
    k = 20

    n_heads = 4
    n_layers = 3
    d_mlp = 32 * 4

    projections = t.randn(
        n_layers * n_heads,
        n_layers * d_mlp,
        10,
        100,
    )
    
    fig = get_fig_head_to_mlp_neuron(projections, quantile, k, n_heads, d_mlp)
    fig.show()
