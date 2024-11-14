import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey


from cardillo.math import (
    ax2skew,
    SE3,
    SE3inv,
    Exp_SE3,
    Log_SE3,
    Exp_SE3_h,
    Log_SE3_H,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    ax2skew_a,
    ax2skew_squared,
)


from ._base import (
    CosseratRod,
    CosseratRodMixed,
    make_CosseratRodConstrained,
)
from ._cross_section import CrossSectionInertias

eye3 = np.eye(3, dtype=float)
eye4 = np.eye(4, dtype=float)
_ax2skew_a = ax2skew_a()


class Quaternion:
    def __init__(self, P, normalize=True):
        self.P = P
        self.p0, self.p = P[0], P[1:]
        self.normalize = normalize

    @property
    def p_tilde(self):
        if not hasattr(self, "_p_tilde"):
            self._p_tilde = ax2skew(self.p)
        return self._p_tilde

    @property
    def p_tilde_square(self):
        if not hasattr(self, "_p_tilde_square"):
            self._p_tilde_square = ax2skew_squared(self.p)
        return self._p_tilde_square

    @property
    def P2(self):
        if not hasattr(self, "_P2"):
            self._P2 = self.P @ self.P
        return self._P2

    @property
    def Exp_SO3_quat(self):
        if not hasattr(self, "_Exp_SO3_quat"):
            if self.normalize:
                self._Exp_SO3_quat = eye3 + (2 / self.P2) * (
                    self.p0 * self.p_tilde + self.p_tilde_square
                )
            else:
                self._Exp_SO3_quat = eye3 + 2 * (
                    self.p0 * self.p_tilde + self.p_tilde_square
                )
        return self._Exp_SO3_quat

    @property
    def Exp_SO3_quat_p(self):
        if not hasattr(self, "_Exp_SO3_quat_p"):
            p_tilde_p = ax2skew_a()

            if self.normalize:
                self._Exp_SO3_quat_p = np.expand_dims(
                    self.p0 * self.p_tilde + self.p_tilde_square, -1
                ) * (-(4 / (self.P2 * self.P2)) * self.P)
                s2 = 2 / self.P2
                self._Exp_SO3_quat_p[:, :, 0] += s2 * self.p_tilde
                self._Exp_SO3_quat_p[:, :, 1:] += (
                    s2 * self.p0 * p_tilde_p
                    + s2 * self.p_tilde.T @ p_tilde_p
                    + (p_tilde_p.T @ (s2 * self.p_tilde.T)).T
                )
            else:
                self._Exp_SO3_quat_p = np.zeros((3, 3, 4), dtype=self.P.dtype)
                self._Exp_SO3_quat_p[:, :, 0] = 2 * self.p_tilde
                self._Exp_SO3_quat_p[:, :, 1:] = (
                    2 * self.p0 * p_tilde_p
                    + 2 * self.p_tilde.T @ p_tilde_p
                    + (p_tilde_p.T @ (2 * self.p_tilde.T)).T
                )
        return self._Exp_SO3_quat_p

    @property
    def T_SO3_quat(self):
        if not hasattr(self, "_T_SO3_quat"):
            if self.normalize:
                self._T_SO3_quat = (2 / self.P2) * np.hstack(
                    (-self.p[:, None], self.p0 * eye3 - self.p_tilde)
                )
            else:
                self._T_SO3_quat = 2 * np.hstack(
                    (-self.p[:, None], self.p0 * eye3 - self.p_tilde)
                )
        return self._T_SO3_quat

    @property
    def T_SO3_inv_quat(self):
        if not hasattr(self, "_T_SO3_inv_quat"):
            if self.normalize:
                self._T_SO3_inv_quat = (0.5 / self.P2) * np.vstack(
                    (-self.p.T, self.p0 * eye3 + self.p_tilde)
                )
            else:
                self._T_SO3_inv_quat = 0.5 * np.vstack(
                    (-self.p.T, self.p0 * eye3 + self.p_tilde)
                )
        return self._T_SO3_inv_quat

    @property
    def T_SO3_quat_P(self):
        if not hasattr(self, "_T_SO3_quat_P"):
            if self.normalize:
                self._T_SO3_quat_P = np.expand_dims(
                    np.hstack((-self.p[:, None], self.p0 * eye3 - ax2skew(self.p))), -1
                ) * (-4 * self.P / (self.P2 * self.P2))
                P22 = 2 / self.P2
                self._T_SO3_quat_P[:, 0, 1:] -= P22 * eye3
                self._T_SO3_quat_P[:, 1:, 0] += P22 * eye3
                self._T_SO3_quat_P[:, 1:, 1:] -= P22 * _ax2skew_a
            else:
                self._T_SO3_quat_P = np.zeros((3, 4, 4), dtype=float)
                self._T_SO3_quat_P[:, 0, 1:] -= 2 * eye3
                self._T_SO3_quat_P[:, 1:, 0] += 2 * eye3
                self._T_SO3_quat_P[:, 1:, 1:] -= 2 * _ax2skew_a
        return self._T_SO3_quat_P

    @property
    def T_SO3_inv_quat_P(self):
        if not hasattr(self, "_T_SO3_inv_quat_P"):
            if self.normalize:
                self._T_SO3_inv_quat_P = np.expand_dims(
                    np.vstack((-self.p.T, self.p0 * eye3 + ax2skew(self.p))), -1
                ) * (-self.P / (self.P2 * self.P2))
                s2 = 0.5 / self.P2
                self._T_SO3_inv_quat_P[0, :, 1:] -= s2 * eye3
                self._T_SO3_inv_quat_P[1:, :, 0] += s2 * eye3
                self._T_SO3_inv_quat_P[1:, :, 1:] += s2 * _ax2skew_a
            else:
                self._T_SO3_inv_quat_P = np.zeros((4, 3, 4), dtype=float)
                self._T_SO3_inv_quat_P[0, :, 1:] = -0.5 * eye3
                self._T_SO3_inv_quat_P[1:, :, 0] = 0.5 * eye3
                self._T_SO3_inv_quat_P[1:, :, 1:] = 0.5 * _ax2skew_a
        return self._T_SO3_inv_quat_P


def make_CosseratRod(interpolation="Quaternion", mixed=True, constraints=None):
    """Rod factory that returns Petrov-Galerkin Cosserat rod classes.

    Parameters
    ----------
    interpolation : str
        Chose interpolation functions
        Quaternion: nodal positions and non-quaternions are interpolated by
        Lagrangian polynomials.
        SE3 : nodal positions and orientations are interpolated respecting the
        SE(3) structure, resulting in a constant strain element.
        R12 : nodal positions anr orientations are interpolated by Lagrangian
        polynomials.
    mixed : bool
        Boolean whether internal forces and couples obtain an independent field.
        Works only in combination with quadratic strain energy functions.
    constraints : array_like
        Set of numbers between 0 and 5 to indicate which strain is constrained.
        0 : Gamma_1 dilatation
        1 : Gamma_2 shear in e_y^K-direction.
        2 : Gamma_3 shear in e_z^K-direction.
        3 : kappa_1 twist.
        4 : kappa_2 flexure around e_y^K-direction.
        5 : kappa_3 flexure around e_z^K-direction.

    Returns:
    --------
    out : CosseratRod class
        Returns CosseratRod_Quat, CosseratRod_SE3 or CosseratRod_R12 class.

    """
    if constraints is not None:
        if not (
            (np.array(constraints) >= 0).all() & (np.array(constraints) <= 5).all()
        ):
            raise ValueError("constraint values must between 0 and 5")

    if interpolation == "Quaternion":
        return make_CosseratRod_Quat(mixed=mixed, constraints=constraints)
    elif interpolation == "SE3":
        return make_CosseratRod_SE3(mixed=mixed, constraints=constraints)
    elif interpolation == "R12":
        return make_CosseratRod_R12(mixed=mixed, constraints=constraints)
    else:
        raise NotImplementedError(
            "This kind of interpolation function has not been implemented."
        )


def make_CosseratRod_Quat(mixed=True, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_Quat(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            polynomial_degree=2,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
        ):
            nquadrature = polynomial_degree
            nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

            if not reduced_integration:
                import warnings

                warnings.warn("Quaternion interpolation: Full integration is used!")
                nquadrature = nquadrature_dyn

            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=polynomial_degree,
                nquadrature=nquadrature,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=nquadrature_dyn,
                cross_section_inertias=cross_section_inertias,
            )

        @staticmethod
        def straight_initial_configuration(
            polynomial_degree,
            basis,
            nelement,
            L,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
            v_P0=np.zeros(3, dtype=float),
            B_omega_IK0=np.zeros(3, dtype=float),
        ):
            return CosseratRod.straight_initial_configuration(
                polynomial_degree,
                polynomial_degree,
                basis,
                basis,
                nelement,
                L,
                r_OP0,
                A_IB0,
                v_P0,
                B_omega_IK0,
            )

        @cachedmethod(
            lambda self: self._eval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _eval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate
            r_OP_nodes = qe[self.nodalDOF_element_r]
            r_OP = N @ r_OP_nodes
            r_OP_xi = N_xi @ r_OP_nodes

            p_nodes = qe[self.nodalDOF_element_p]
            p_xi = N_xi @ p_nodes
            p = N @ p_nodes

            p = Quaternion(p, normalize=True)
            # transformation matrix
            A_IB = p.Exp_SO3_quat

            # dilatation and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi

            # curvature, Rucker2018 (17)
            B_Kappa_bar = p.T_SO3_quat @ p_xi

            return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

        @cachedmethod(
            lambda self: self._eval_all_cache,
            key=lambda self, qe, xis, Ns, N_xis: hashkey(*qe, *xis),
        )
        def _eval_all(self, qe, xis, Ns, N_xis):
            # interpolate
            r_OP_nodes = qe[self.nodalDOF_element_r]
            r_OPs = Ns @ r_OP_nodes
            r_OP_xis = N_xis @ r_OP_nodes

            p_nodes = qe[self.nodalDOF_element_p]
            p_xis = N_xis @ p_nodes
            ps = Ns @ p_nodes

            n = len(ps)
            A_IBs = np.empty((n, 3, 3))
            B_Gamma_bars = np.empty((n, 3))
            B_Kappa_bars = np.empty((n, 3))
            for i, p in enumerate(ps):
                p = Quaternion(p, normalize=True)
                # transformation matrix
                A_IBs[i] = p.Exp_SO3_quat

                # dilatation and shear strains
                B_Gamma_bars[i] = A_IBs[i].T @ r_OP_xis[i]

                # curvature, Rucker2018 (17)
                B_Kappa_bars[i] = p.T_SO3_quat @ p_xis[i]
            # B_Gamma_bars = (np.swapaxes(A_IBs, 1, 2) @ np.expand_dims(r_OP_xis, -1)).squeeze()
            return r_OPs, A_IBs, B_Gamma_bars, B_Kappa_bars

        @cachedmethod(
            lambda self: self._deval_all_cache,
            key=lambda self, qe, xis, Ns, N_xis: hashkey(*qe, *xis),
        )
        def _deval_all(self, qe, xis, Ns, N_xis):
            # interpolate
            n = len(Ns)
            r_OP_qes = np.zeros((n, 3, self.nq_element), dtype=qe.dtype)
            r_OP_xi_qes = np.zeros((n, 3, self.nq_element), dtype=qe.dtype)

            p_qes = np.zeros((n, 4, self.nq_element), dtype=qe.dtype)
            p_xi_qes = np.zeros((n, 4, self.nq_element), dtype=qe.dtype)

            r_OP_nodes = qe[self.nodalDOF_element_r]
            r_OPs = Ns @ r_OP_nodes
            r_OP_qes[:, :, self.nodalDOF_element_r.T] = np.expand_dims(
                eye3, 2
            ) * np.expand_dims(Ns, [1, 2])

            r_OP_xis = N_xis @ r_OP_nodes
            r_OP_xi_qes[:, :, self.nodalDOF_element_r.T] = np.expand_dims(
                eye3, 2
            ) * np.expand_dims(N_xis, [1, 2])

            p_nodes = qe[self.nodalDOF_element_p]
            ps = Ns @ p_nodes
            p_qes[:, :, self.nodalDOF_element_p.T] = np.expand_dims(
                eye4, 2
            ) * np.expand_dims(Ns, [1, 2])

            p_xis = N_xis @ p_nodes
            p_xi_qes[:, :, self.nodalDOF_element_p.T] = np.expand_dims(
                eye4, 2
            ) * np.expand_dims(N_xis, [1, 2])

            A_IBs = np.empty((n, 3, 3))
            A_IB_qes = np.empty((n, 3, 3, self.nq_element))
            B_Gamma_bars = np.empty((n, 3))
            B_Gamma_bar_qes = np.empty((n, 3, self.nq_element))
            B_Kappa_bars = np.empty((n, 3))
            B_Kappa_bar_qes = np.empty((n, 3, self.nq_element))
            for i, p in enumerate(ps):
                p = Quaternion(p, normalize=True)
                # transformation matrix
                A_IBs[i] = p.Exp_SO3_quat

                # derivative w.r.t. generalized coordinates
                A_IB_qes[i] = p.Exp_SO3_quat_p @ p_qes[i]

                # axial and shear strains
                B_Gamma_bars[i] = A_IBs[i].T @ r_OP_xis[i]
                B_Gamma_bar_qes[i] = (
                    np.moveaxis(A_IB_qes[i], 0, -1) @ r_OP_xis[i]
                    + A_IBs[i].T @ r_OP_xi_qes[i]
                )
                # curvature, Rucker2018 (17)
                T = p.T_SO3_quat
                B_Kappa_bars[i] = T @ p_xis[i]

                # B_Kappa_bar_qe = approx_fprime(qe, lambda qe: self._eval(qe, xi)[3])
                B_Kappa_bar_qes[i] = (
                    p_xis[i] @ p.T_SO3_quat_P @ p_qes[i] + T @ p_xi_qes[i]
                )

            return (
                r_OPs,
                A_IBs,
                B_Gamma_bars,
                B_Kappa_bars,
                r_OP_qes,
                A_IB_qes,
                B_Gamma_bar_qes,
                B_Kappa_bar_qes,
            )

        @cachedmethod(
            lambda self: self._deval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _deval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate
            r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)

            p_qe = np.zeros((4, self.nq_element), dtype=qe.dtype)
            p_xi_qe = np.zeros((4, self.nq_element), dtype=qe.dtype)

            r_OP_nodes = qe[self.nodalDOF_element_r]
            r_OP = N @ r_OP_nodes
            r_OP_qe[:, self.nodalDOF_element_r.T] = np.expand_dims(eye3, 2) * N

            r_OP_xi = N_xi @ r_OP_nodes
            r_OP_xi_qe[:, self.nodalDOF_element_r.T] = np.expand_dims(eye3, 2) * N_xi

            p_nodes = qe[self.nodalDOF_element_p]
            p = N @ p_nodes
            p_qe[:, self.nodalDOF_element_p.T] = np.expand_dims(eye4, 2) * N

            p_xi = N_xi @ p_nodes
            p_xi_qe[:, self.nodalDOF_element_p.T] = np.expand_dims(eye4, 2) * N_xi

            p = Quaternion(p, normalize=True)
            # transformation matrix
            A_IB = p.Exp_SO3_quat

            # derivative w.r.t. generalized coordinates
            A_IB_qe = p.Exp_SO3_quat_p @ p_qe

            # axial and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi
            B_Gamma_bar_qe = np.moveaxis(A_IB_qe, 0, -1) @ r_OP_xi + A_IB.T @ r_OP_xi_qe
            # curvature, Rucker2018 (17)
            T = p.T_SO3_quat
            B_Kappa_bar = T @ p_xi

            # B_Kappa_bar_qe = approx_fprime(qe, lambda qe: self._eval(qe, xi)[3])
            B_Kappa_bar_qe = p_xi @ p.T_SO3_quat_P @ p_qe + T @ p_xi_qe

            return (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            )

        def A_IB(self, t, qe, xi):
            N, N_xi = self.basis_functions_r(xi)
            return self._eval(qe, xi, N, N_xi)[1]
            # # evaluate shape functions
            # N_p, _ = self.basis_functions_p(xi)

            # # interpolate orientation
            # A_IB = np.zeros((3, 3), dtype=q.dtype)
            # for node in range(self.nnodes_element_p):
            #     A_IB += N_p[node] * self.Exp_SO3_quat(
            #         q[self.nodalDOF_element_p[node]]
            #     )

            # return A_IB

        def A_IB_q(self, t, qe, xi):
            # return approx_fprime(q, lambda q: self.A_IB(t, q, xi))
            N, N_xi = self.basis_functions_r(xi)
            return self._deval(qe, xi, N, N_xi)[5]

    return CosseratRod_Quat


def make_CosseratRod_SE3(mixed=True, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_SE3(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
            **kwargs,
        ):
            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=1,
                nquadrature=1 if reduced_integration else 2,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=2,
                cross_section_inertias=cross_section_inertias,
            )

        @staticmethod
        def straight_configuration(
            nelement,
            L,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
            **kwargs,
        ):
            return CosseratRod.straight_configuration(nelement, L, 1, r_OP0, A_IB0)

        @staticmethod
        def deformed_configuration(
            nelement,
            curve,
            dcurve,
            ddcurve,
            xi1,
            polynomial_degree=1,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
        ):
            return CosseratRod.deformed_configuration(
                nelement,
                curve,
                dcurve,
                ddcurve,
                xi1,
                polynomial_degree=1,
                r_OP0=r_OP0,
                A_IB0=A_IB0,
            )

        @staticmethod
        def straight_initial_configuration(
            nelement,
            L,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
            v_P=np.zeros(3, dtype=float),
            B_omega_IK=np.zeros(3, dtype=float),
            **kwargs,
        ):
            return CosseratRod.straight_initial_configuration(
                nelement,
                L,
                polynomial_degree=1,
                r_OP0=r_OP0,
                A_IB0=A_IB0,
                v_P0=v_P,
                B_omega_IK0=B_omega_IK,
            )

        @cachedmethod(
            lambda self: self._eval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _eval(self, qe, xi, N, N_xi):
            # nodal unknowns
            r_OP0, r_OP1 = qe[self.nodalDOF_element_r]
            p0, p1 = qe[self.nodalDOF_element_p]

            # nodal transformations
            A_IB0 = Exp_SO3_quat(p0)
            A_IB1 = Exp_SO3_quat(p1)
            H_IK0 = SE3(A_IB0, r_OP0)
            H_IK1 = SE3(A_IB1, r_OP1)

            # inverse transformation of first node
            H_IK0_inv = SE3inv(H_IK0)

            # compute relative transformation
            H_K0K1 = H_IK0_inv @ H_IK1

            # compute relative screw
            h_K0K1 = Log_SE3(H_K0K1)

            # find element number containing xi
            el = self.element_number(xi)

            # get element interval
            xi0, xi1 = self.knot_vector_r.element_interval(el)

            # second linear Lagrange shape function
            N1_xi = 1.0 / (xi1 - xi0)
            N1 = (xi - xi0) * N1_xi

            # relative interpolation of local se(3) objects
            h_local = N1 * h_K0K1
            h_local_xi = N1_xi * h_K0K1

            # composition of reference and local transformation
            H_local = Exp_SE3(h_local)
            H_IK = H_IK0 @ H_local

            # extract centerline and transformation matrix
            A_IB = H_IK[:3, :3]
            r_OP = H_IK[:3, 3]

            # extract strains
            B_Gamma_bar = h_local_xi[:3]
            B_Kappa_bar = h_local_xi[3:]

            return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

        @cachedmethod(
            lambda self: self._deval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _deval(self, qe, xi, N, N_xi):
            # extract nodal screws
            nodalDOF0 = np.concatenate(
                (self.nodalDOF_element_r[0], self.nodalDOF_element_p[0])
            )
            nodalDOF1 = np.concatenate(
                (self.nodalDOF_element_r[1], self.nodalDOF_element_p[1])
            )
            h0 = qe[nodalDOF0]
            r_OP0 = h0[:3]
            p0 = h0[3:]
            h1 = qe[nodalDOF1]
            r_OP1 = h1[:3]
            p1 = h1[3:]

            # nodal transformations
            A_IB0 = Exp_SO3_quat(p0)
            A_IB1 = Exp_SO3_quat(p1)
            H_IK0 = SE3(A_IB0, r_OP0)
            H_IK1 = SE3(A_IB1, r_OP1)
            A_IB0_p0 = Exp_SO3_quat_p(p0)
            A_IB1_p1 = Exp_SO3_quat_p(p1)

            H_IK0_h0 = np.zeros((4, 4, 7), dtype=float)
            H_IK0_h0[:3, :3, 3:] = A_IB0_p0
            H_IK0_h0[:3, 3, :3] = eye3
            H_IB1_h1 = np.zeros((4, 4, 7), dtype=float)
            H_IB1_h1[:3, :3, 3:] = A_IB1_p1
            H_IB1_h1[:3, 3, :3] = eye3

            # inverse transformation of first node
            H_IK0_inv = SE3inv(H_IK0)
            H_IK0_inv_h0 = np.zeros((4, 4, 7), dtype=float)
            H_IK0_inv_h0[:3, :3, 3:] = A_IB0_p0.transpose(1, 0, 2)
            H_IK0_inv_h0[:3, 3, 3:] = -np.einsum("k,kij->ij", r_OP0, A_IB0_p0)
            H_IK0_inv_h0[:3, 3, :3] = -A_IB0.T

            # compute relative transformation
            H_K0K1 = H_IK0_inv @ H_IK1
            H_K0B1_h0 = np.einsum("ilk,lj->ijk", H_IK0_inv_h0, H_IK1)
            H_K0B1_h1 = np.einsum("il,ljk->ijk", H_IK0_inv, H_IB1_h1)

            # compute relative screw
            h_K0K1 = Log_SE3(H_K0K1)
            h_K0B1_HK0K1 = Log_SE3_H(H_K0K1)
            h_K0B1_h0 = np.einsum("ikl,klj->ij", h_K0B1_HK0K1, H_K0B1_h0)
            h_K0B1_h1 = np.einsum("ikl,klj->ij", h_K0B1_HK0K1, H_K0B1_h1)

            # find element number containing xi
            el = self.element_number(xi)

            # get element interval
            xi0, xi1 = self.knot_vector_r.element_interval(el)

            # second linear Lagrange shape function
            N1_xi = 1.0 / (xi1 - xi0)
            N1 = (xi - xi0) * N1_xi

            # relative interpolation of local se(3) objects
            h_local = N1 * h_K0K1
            h_local_xi = N1_xi * h_K0K1
            h_local_h0 = N1 * h_K0B1_h0
            h_local_h1 = N1 * h_K0B1_h1
            h_local_xi_h0 = N1_xi * h_K0B1_h0
            h_local_xi_h1 = N1_xi * h_K0B1_h1

            # composition of reference and local transformation
            H_local = Exp_SE3(h_local)
            H_local_h = Exp_SE3_h(h_local)
            H_local_h0 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h0)
            H_local_h1 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h1)
            H_IK = H_IK0 @ H_local
            H_IB_h0 = np.einsum("ilk,lj", H_IK0_h0, H_local) + np.einsum(
                "il,ljk->ijk", H_IK0, H_local_h0
            )
            H_IB_h1 = np.einsum("il,ljk->ijk", H_IK0, H_local_h1)

            # extract centerline and transformation matrix
            A_IB = H_IK[:3, :3]
            r_OP = H_IK[:3, 3]
            A_IB_qe = np.zeros((3, 3, self.nq_element), dtype=float)
            A_IB_qe[:, :, nodalDOF0] = H_IB_h0[:3, :3]
            A_IB_qe[:, :, nodalDOF1] = H_IB_h1[:3, :3]
            r_OP_qe = np.zeros((3, self.nq_element), dtype=float)
            r_OP_qe[:, nodalDOF0] = H_IB_h0[:3, 3]
            r_OP_qe[:, nodalDOF1] = H_IB_h1[:3, 3]

            # extract strains
            B_Gamma_bar = h_local_xi[:3]
            B_Kappa_bar = h_local_xi[3:]
            B_Gamma_bar_qe = np.zeros((3, self.nq_element), dtype=float)
            B_Gamma_bar_qe[:, nodalDOF0] = h_local_xi_h0[:3]
            B_Gamma_bar_qe[:, nodalDOF1] = h_local_xi_h1[:3]
            B_Kappa_bar_qe = np.zeros((3, self.nq_element), dtype=float)
            B_Kappa_bar_qe[:, nodalDOF0] = h_local_xi_h0[3:]
            B_Kappa_bar_qe[:, nodalDOF1] = h_local_xi_h1[3:]

            return (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            )

        def A_IB(self, t, qe, xi):
            N, N_xi = None, None
            return self._eval(qe, xi, N, N_xi)[1]

        def A_IB_q(self, t, qe, xi):
            N, N_xi = None, None
            return self._deval(qe, xi, N, N_xi)[5]

    return CosseratRod_SE3


def make_CosseratRod_R12(mixed=True, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_R12(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            polynomial_degree=2,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
        ):
            nquadrature = polynomial_degree
            nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

            if not reduced_integration:
                import warnings

                warnings.warn("'R12_PetrovGalerkin': Full integration is used!")
                nquadrature = nquadrature_dyn

            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=polynomial_degree,
                nquadrature=nquadrature,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=nquadrature_dyn,
                cross_section_inertias=cross_section_inertias,
            )

        # returns interpolated positions, orientations and strains at xi in [0,1]
        @cachedmethod(
            lambda self: self._eval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _eval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate position and tangent vector
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                r_OP_node = qe[self.nodalDOF_element_r[node]]
                r_OP += N[node] * r_OP_node
                r_OP_xi += N_xi[node] * r_OP_node

            # interpolate transformation matrix and its derivative
            A_IB = np.zeros((3, 3), dtype=qe.dtype)
            A_IB_xi = np.zeros((3, 3), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                A_IB_node = Exp_SO3_quat(qe[self.nodalDOF_element_p[node]])
                A_IB += N[node] * A_IB_node
                A_IB_xi += N_xi[node] * A_IB_node

            # axial and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi

            # torsional and flexural strains
            d1, d2, d3 = A_IB.T
            d1_xi, d2_xi, d3_xi = A_IB_xi.T
            B_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )

            return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

        @cachedmethod(
            lambda self: self._deval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _deval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate position and tangent vector + their derivatives
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                nodalDOF_r = self.nodalDOF_element_r[node]
                r_OP_node = qe[nodalDOF_r]

                r_OP += N[node] * r_OP_node
                r_OP_qe[:, nodalDOF_r] += N[node] * eye3

                r_OP_xi += N_xi[node] * r_OP_node
                r_OP_xi_qe[:, nodalDOF_r] += N_xi[node] * eye3

            # interpolate transformation matrix and its derivative + their derivatives
            A_IB = np.zeros((3, 3), dtype=qe.dtype)
            A_IB_xi = np.zeros((3, 3), dtype=qe.dtype)
            A_IB_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            A_IB_xi_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                nodalDOF_p = self.nodalDOF_element_p[node]
                p_node = qe[nodalDOF_p]
                A_IB_node = Exp_SO3_quat(p_node)
                A_IB_q_node = Exp_SO3_quat_p(p_node)

                A_IB += N[node] * A_IB_node
                A_IB_qe[:, :, nodalDOF_p] += N[node] * A_IB_q_node

                A_IB_xi += N_xi[node] * A_IB_node
                A_IB_xi_qe[:, :, nodalDOF_p] += N_xi[node] * A_IB_q_node

            # extract directors
            d1, d2, d3 = A_IB.T
            d1_xi, d2_xi, d3_xi = A_IB_xi.T
            d1_qe, d2_qe, d3_qe = A_IB_qe.transpose(1, 0, 2)
            d1_xi_qe, d2_xi_qe, d3_xi_qe = A_IB_xi_qe.transpose(1, 0, 2)

            # axial and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi
            B_Gamma_bar_qe = np.einsum("k,kij", r_OP_xi, A_IB_qe) + A_IB.T @ r_OP_xi_qe

            # torsional and flexural strains
            B_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )
            B_Kappa_bar_qe = np.array(
                [
                    0.5
                    * (d3 @ d2_xi_qe + d2_xi @ d3_qe - d2 @ d3_xi_qe - d3_xi @ d2_qe),
                    0.5
                    * (d1 @ d3_xi_qe + d3_xi @ d1_qe - d3 @ d1_xi_qe - d1_xi @ d3_qe),
                    0.5
                    * (d2 @ d1_xi_qe + d1_xi @ d2_qe - d1 @ d2_xi_qe - d2_xi @ d1_qe),
                ]
            )

            return (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            )

        def A_IB(self, t, qe, xi):
            # evaluate shape functions
            N_p, _ = self.basis_functions_p(xi)

            # interpolate orientation
            A_IB = np.zeros((3, 3), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                A_IB += N_p[node] * Exp_SO3_quat(qe[self.nodalDOF_element_p[node]])

            return A_IB

        def A_IB_q(self, t, qe, xi):
            # evaluate shape functions
            N_p, _ = self.basis_functions_p(xi)

            # interpolate centerline position and orientation
            A_IB_q = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                nodalDOF_p = self.nodalDOF_element_p[node]
                A_IB_q[:, :, nodalDOF_p] += N_p[node] * Exp_SO3_quat_p(qe[nodalDOF_p])

            return A_IB_q

    return CosseratRod_R12
