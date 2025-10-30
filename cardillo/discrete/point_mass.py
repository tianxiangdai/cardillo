import numpy as np
from vtk import VTK_VERTEX
from ._base import PositionKinematics


class PointMass(PositionKinematics):
    def __init__(
        self,
        mass,
        q0=None,
        u0=None,
        name="point_mass",
    ):
        """Point mass parametrized by center of mass in inertial basis.

        Parameters
        ----------
        mass : float
            Mass of point mass.
        q0 : np.array(3)
            Initial position coordinates at time t0.
        u0 : np.array(3)
            Initial velocity coordinates at time t0.
        name : str
            Name of point mass.
        """
        self.nq = 3
        self.nu = 3

        self.q0 = np.zeros(self.nq) if q0 is None else np.asarray(q0)
        self.u0 = np.zeros(self.nu) if u0 is None else np.asarray(u0)
        assert self.q0.size == self.nq
        assert self.u0.size == self.nu

        self.mass = mass
        self.__M = mass * np.eye(3)

        self.name = name

    #####################
    # kinetic energy
    #####################
    def E_kin(self, t, q, u):
        return 0.5 * self.mass * np.dot(u, u)

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.__M

    ########
    # export
    ########
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(v=[vel])
        return points, cells, None, cell_data
