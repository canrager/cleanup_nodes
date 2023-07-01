import einops
from jaxtyping import Float
from torch import Tensor


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

    Mathematically: cos_sim(A, B) = (A•B) / (||B||)
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
