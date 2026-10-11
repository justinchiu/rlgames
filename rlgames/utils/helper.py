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
    """Elementwise `jnp.where` over two pytrees with the same structure."""
    return jax.tree.map(lambda x, y: jnp.where(pred, x, y), on_true, on_false)
