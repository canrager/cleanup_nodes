import torch
import einops

from typing import List, Tuple, Optional, Callable
from jaxtyping import Float
from torch import Tensor

from tqdm.auto import tqdm, trange


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


def projection_ratio_cartesian(
    cleaner_vectors: Float[Tensor, "dim0_arg0 ... d_model"],
    writer_vectors: Float[Tensor, "dim0_arg1 ... d_model"],
) -> Float[Tensor, "..."]:
    """
    Same as `projection_ratio`, but takes the Cartesian product of the
    first dimension of the input tensors.
    """
    return einops.einsum(
        cleaner_vectors,
        writer_vectors / writer_vectors.norm(dim=-1, keepdim=True).pow(2),
        "dim0_arg0 ... d_model, dim0_arg1 ... d_model -> dim0_arg0 dim0_arg1 ...",
    )


def cos_sim(
    cleaner_vectors: Float[Tensor, "... d_model"],
    writer_vectors: Float[Tensor, "... d_model"],
) -> Float[Tensor, "..."]:
    """
    Works element-wise on the last dimension of the input tensors.

    Mathematically: cos_sim(A, B) = (A•B) / (||A|| * ||B||)
    """
    return einops.einsum(
        cleaner_vectors / cleaner_vectors.norm(dim=-1, keepdim=True),
        writer_vectors / writer_vectors.norm(dim=-1, keepdim=True),
        "... d_model, ... d_model -> ...",
    )


def cos_sim_cartesian(
    cleaner_vectors: Float[Tensor, "dim0_arg0 ... d_model"],
    writer_vectors: Float[Tensor, "dim0_arg1 ... d_model"],
) -> Float[Tensor, "..."]:
    """
    Same as `cos_sim`, but takes the Cartesian product of the
    first dimension of the input tensors.
    """
    return einops.einsum(
        cleaner_vectors / cleaner_vectors.norm(dim=-1, keepdim=True),
        writer_vectors / writer_vectors.norm(dim=-1, keepdim=True),
        "dim0_arg0 ... d_model, dim0_arg1 ... d_model -> dim0_arg0 dim0_arg1 ...",
    )


def projection_value(
    cleaner_vectors: Float[Tensor, "... d_model"],
    writer_vectors: Float[Tensor, "... d_model"],
) -> Float[Tensor, "..."]:
    """
    Works element-wise on the last dimension of the input tensors.

    Mathematically: projection_value(A, B) = (A•B) / ||B||
    """
    return einops.einsum(
        cleaner_vectors,
        writer_vectors / writer_vectors.norm(dim=-1, keepdim=True),
        "... d_model, ... d_model -> ...",
    )


def projection_value_cartesian(
    cleaner_vectors: Float[Tensor, "dim0_arg0 ... d_model"],
    writer_vectors: Float[Tensor, "dim0_arg1 ... d_model"],
) -> Float[Tensor, "..."]:
    """
    Same as `projection_value`, but takes the Cartesian product of the
    first dimension of the input tensors.
    """
    return einops.einsum(
        cleaner_vectors,
        writer_vectors / writer_vectors.norm(dim=-1, keepdim=True),
        "dim0_arg0 ... d_model, dim0_arg1 ... d_model -> dim0_arg0 dim0_arg1 ...",
    )


def get_logit_diff_function(
    model,
    correct_token_id: int,
    incorrect_token_id: int,
) -> Callable:
    """
    Args:
        model: a HookedTransformer from Transformer Lens
        correct_token_id: the token id of the correct token
        incorrect_token_id: the token id of the incorrect token

    Returns:
        A function that takes a tensor of residuals and returns the
        difference in logits between the correct and incorrect token.
    """
    logit_diff_direction = (
        model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id]
    )

    logit_diff_bias = 0
    if hasattr(model, "b_U"):
        logit_diff_bias = model.b_U[correct_token_id] - model.b_U[incorrect_token_id]

    def calc_logit_diff(
        resid_final: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "..."]:
        return resid_final @ logit_diff_direction + logit_diff_bias

    return calc_logit_diff


def scale_embeddings(
    model,
    token_ids_to_scale: Optional[List[int]] = None,
    device="cpu",
) -> Tuple[Float[Tensor, "d_vocab d_model"], Float[Tensor, "n_ctx d_model"]]:
    """
    Scales the token and positional embeddings of a model.

    Taken from Jett's notebook:
    https://github.com/jettjaniak/research/blob/70f1a09910953a6c909b876ad60bb5c350ac9cfc/014-p-to-t-tok-stats.ipynb

    Args:
        model: a HookedTransformer from Transformer Lens
        token_ids_to_scale: a list of token IDs to apply scaling to. If `None`,
            then all token IDs are scaled.

    Returns:
        A tuple of (scaled_token_embeddings, scaled_positional_embeddings)
    """
    W_E = model.W_E.data  # token embeddings, (d_vocab, d_model)
    W_pos = model.W_pos.data  # positional embeddings, (n_ctx, d_model)

    if token_ids_to_scale is None:
        token_ids_to_scale = list(range(model.W_E.shape[0]))

    def decompose_resid0():
        b_T_ = W_E[token_ids_to_scale].mean(dim=0, keepdim=True)
        b_P_ = W_pos[1:].mean(dim=0, keepdim=True)
        b_TP_ = b_T_ + b_P_
        T_ = torch.zeros_like(W_E)
        T_[token_ids_to_scale] = W_E[token_ids_to_scale] - b_T_
        P_ = torch.zeros_like(W_pos)
        P_[1:] = W_pos[1:] - b_P_
        return T_, P_, b_TP_

    def layer_norm_scale(x):
        x2 = x.pow(2)
        x2mean = x2.mean(-1, keepdim=True)
        x2mean_eps = x2mean + model.cfg.eps
        return x2mean_eps.sqrt()

    def get_T_ln_scale(T_, P_, b_TP_):
        print("computing T LN scaling")
        T_ln_scale = torch.ones(T_.shape[0], 1).to(device)
        for t in tqdm(token_ids_to_scale):
            combined_embed = T_[t : t + 1] + P_[1:] + b_TP_
            combined_embeds_norms = layer_norm_scale(combined_embed)
            T_ln_scale[t] = combined_embeds_norms.median()
        return T_ln_scale

    def get_P_ln_scale(T_, P_, b_TP_):
        print("computing P LN scaling")
        P_ln_scale = torch.ones(model.cfg.n_ctx, 1).to(device)
        for p in trange(1, model.cfg.n_ctx):
            combined_embed = P_[p : p + 1] + T_[token_ids_to_scale] + b_TP_
            combined_embeds_norms = layer_norm_scale(combined_embed)
            P_ln_scale[p] = combined_embeds_norms.median()
        return P_ln_scale

    T_, P_, b_TP_ = decompose_resid0()

    T_ln_scale = get_T_ln_scale(T_, P_, b_TP_)
    P_ln_scale = get_P_ln_scale(T_, P_, b_TP_)

    scaled_token_embeddings = T_ / T_ln_scale
    scaled_positional_embeddings = P_ / P_ln_scale

    return scaled_token_embeddings, scaled_positional_embeddings
