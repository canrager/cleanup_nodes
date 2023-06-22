# %%
import itertools
import numpy as np
import torch as t

import einops
from jaxtyping import Float

import plotly.express as px



def _neuron_index_to_name(neuron_index: int, d_mlp: int) -> str:
    layer, neuron = divmod(neuron_index, d_mlp)
    return f"N{layer}.{neuron}"


def _head_index_to_name(head_index: int, n_heads: int) -> str:
    layer, head = divmod(head_index, n_heads)
    return f"H{layer}.{head}"


def get_fig_head_mlp_neuron(
    projections: Float[t.Tensor, "head neuron prompt pos"],
    quantile: int,
    k: int,
    n_heads: int,
    d_mlp: int,
) -> None:
    """
    Plot the top k neurons for each head, where the top k neurons are
    determined by the quantile of the projection matrix.
    """

    # Get quantiles
    projections_quantile = einops.rearrange(
        projections,
        "head neuron prompt pos -> head neuron (prompt pos)",
    ).quantile(quantile, dim=-1)

    # Get top k neurons for each head
    projections_topk = projections_quantile.topk(k, dim=-1, largest=False)

    # Create head names and neuron names for hover text
    nrows, ncols = projections_topk.values.shape
    head_names = [_head_index_to_name(i, n_heads) for i in range(projections_topk.values.shape[0])]
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
        # hoverinfo="text",
        # text=neuron_names,
    )

    # Add hover text
    hovertemplate = (
        'Metric Name: %{z}<br><br>'
        'Head Name: %{y}<br>'
        'Neuron Name: %{customdata}<br>'
        'Neuron Rank: %{x}<br>'
    )

    fig.update(
        data=[{
            'hovertemplate': hovertemplate,
            'customdata': neuron_names,
        }]
    )

    # fig.show()
    return fig
