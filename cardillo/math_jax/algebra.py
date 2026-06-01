import jax.numpy as jnp
from jax import jit, vmap


@jit
def ax2skew(a: jnp.ndarray) -> jnp.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    # fmt: off
    return jnp.array([[0,    -a[2], a[1] ],
                      [a[2],  0,    -a[0]],
                      [-a[1], a[0], 0    ]], dtype=jnp.float64)
    # fmt: on


ax2skew_batch = jit(vmap(ax2skew))


@jit
def ax2skew_squared(a: jnp.ndarray) -> jnp.ndarray:
    """Computes the product of a skew-symmetric matrix with itself from a given axial vector."""
    a1, a2, a3 = a
    # fmt: off
    return jnp.array([
        [-a2**2 - a3**2,              a1 * a2,              a1 * a3],
        [             a2 * a1, -a1**2 - a3**2,              a2 * a3],
        [             a3 * a1,              a3 * a2, -a1**2 - a2**2],
    ], dtype=jnp.float64)
    # fmt: on
