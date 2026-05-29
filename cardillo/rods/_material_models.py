import numpy as np

# from cardillo.math import norm, approx_fprime
from cardillo.math.algebra import norm


class Simo1986:
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable rod with quadratic strain energy
        function found in Simo1986 (2.8), (2.9) and (2.10).

        Parameters
        ----------
        Ei : np.ndarray (3,)
            E0: dilatational stiffness, i.e., rigidity with resepct to volumetric change.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray (3,)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        """

        self._variable = callable(Ei) or callable(Fi)
        self.Ei = Ei
        self.Fi = Fi

        self.C_n = (
            (lambda xi: np.diag(self.Ei(xi))) if self._variable else np.diag(self.Ei)
        )
        self.C_m = (
            (lambda xi: np.diag(self.Fi(xi))) if self._variable else np.diag(self.Fi)
        )

        self.C_n_inv = (
            (lambda xi: np.linalg.inv(self.C_n(xi)))
            if self._variable
            else np.linalg.inv(self.C_n)
        )
        self.C_m_inv = (
            (lambda xi: np.linalg.inv(self.C_m(xi)))
            if self._variable
            else np.linalg.inv(self.C_m)
        )

    def potential(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        dG = B_Gamma - B_Gamma0
        dK = B_Kappa - B_Kappa0
        return 0.5 * dG @ self.C_n @ dG + 0.5 * dK @ self.C_m @ dK

    def complementary_potential(self, B_n, B_m):
        if self._variable:
            raise NotImplementedError
        return 0.5 * B_n @ self.C_n_inv @ B_n + 0.5 * B_m @ self.C_m_inv @ B_m

    def B_n(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        dG = B_Gamma - B_Gamma0
        return self.C_n @ dG

    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        dK = B_Kappa - B_Kappa0
        return self.C_m @ dK

    def B_n_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        return self.C_n

    def B_n_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        if self._variable:
            raise NotImplementedError
        return self.C_m


