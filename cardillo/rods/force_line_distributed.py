import numpy as np
from jax import jit, vmap
from jax import numpy as jnp



class Force_line_distributed:
    def __init__(self, force, rod):
        r"""Line distributed dead load for rods

        Parameters
        ----------
        force : np.ndarray (3,)
            Force w.r.t. inertial I-basis as a callable function in time t and
            rod position xi.
        rod : CosseratRod
            Cosserat rod from Cardillo.

        """
        if not callable(force):
            _force = lambda t, xi: force
        else:
            _force = force
        self.rod = rod
        self._h_weights = (
            np.pad(rod.L_els, (1, 0)) + np.pad(rod.L_els, (0, 1))
        ) / 2

        def h_node(t, xi, weight):
            return jnp.pad(_force(t, xi), (0, 3)) * weight

        self._h_nodes = jit(vmap(h_node, in_axes=(None, 0, 0)))

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF

    #####################
    # equations of motion
    #####################
    def h(self, t, q, u):
        return np.asarray(
            self._h_nodes(t, self.rod.xi_node, self._h_weights)
        ).ravel()

