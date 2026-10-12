import jax
import jax.numpy as jnp


def one_hot(index: jax.Array, n: int) -> jax.Array:
    """One-hot encoding that is all zeros when `index` is negative."""
    return jnp.where(index[..., None] >= 0, jax.nn.one_hot(index, n), 0.0)


def uniform_over(mask: jax.Array) -> jax.Array:
    """Uniform distribution over the True entries of a bool mask."""
    mask = mask.astype(jnp.float32)
    return mask / mask.sum()



def select(pred: jax.Array, on_true, on_false):
    """`jnp.where` over two pytrees with the same structure. `pred` may be batched
    over leading axes; it is broadcast over each leaf's trailing axes."""

    def where(x, y):
        p = jnp.reshape(pred, jnp.shape(pred) + (1,) * (jnp.ndim(x) - jnp.ndim(pred)))
        return jnp.where(p, x, y)

    return jax.tree.map(where, on_true, on_false)
