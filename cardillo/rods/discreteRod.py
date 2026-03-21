import numpy as np
from numpy.lib.stride_tricks import as_strided

import jax
from jax import vmap, jit
from jax import numpy as jnp
from numba import njit

from cachetools import cachedmethod, LRUCache

from cardillo.math_numba import (
    norm,
    cross3,
    ax2skew,
    Log_SO3_quat,
    Exp_SO3_quat,
    Exp_SO3_quat_P,
)
from cardillo import math_jax
from cardillo.rods._base import RodExportBase
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.rods import CrossSectionInertias

from cardillo.math import A_IB_basic
from cardillo.utility.check_time_derivatives import check_time_derivatives

from .sensor import Sensor

jax.config.update("jax_enable_x64", True)

eye3 = jnp.eye(3, dtype=jnp.float64)
zeros3 = jnp.zeros((3, 3))


_nla_c_el = 6  # 6/12


class DiscreteRod(RodExportBase):
    def __init__(
        self,
        cross_section,
        material_model,
        nelement,
        Q,
        *,
        q0=None,
        u0=None,
        cross_section_inertias=CrossSectionInertias(),
        name="discrete_node",
    ):
        super().__init__(cross_section)
        self.material_model = material_model
        self.nelement = nelement
        self.nnode = nelement + 1
        self.name = name
        self.C_n_inv = material_model.C_n_inv
        self.C_m_inv = material_model.C_m_inv

        # total DOFs
        self.nq = 7 * self.nnode
        self.nu = 6 * self.nnode
        self.nla_S = self.nnode
        self.nla_c = self.nelement * _nla_c_el

        self.q0 = Q if q0 is None else np.asarray(q0)
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        self.xis = np.linspace(0, 1, self.nnode)

        # slices of DOFs
        self.elDOF = [slice(7 * el, 7 * (el + 2)) for el in range(self.nelement)]
        self.elDOF_u = [slice(6 * el, 6 * (el + 2)) for el in range(self.nelement)]
        self.elDOF_la_c = [
            slice(_nla_c_el * el, _nla_c_el * (el + 1)) for el in range(self.nelement)
        ]
        self.nodalDOF = [slice(7 * n, 7 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_r = [slice(7 * n, 7 * n + 3) for n in range(self.nnode)]
        self.nodalDOF_p = [slice(7 * n + 3, 7 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_u = [slice(6 * n, 6 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_r_u = [slice(6 * n, 6 * n + 3) for n in range(self.nnode)]
        self.nodalDOF_p_u = [slice(6 * n + 3, 6 * (n + 1)) for n in range(self.nnode)]

        self.set_reference_strains(Q)

        self.constant_mass_matrix = True
        self.__M = CooMatrix((self.nu, self.nu))
        self._B_Theta_C = []
        for n in range(self.nnode):
            if n == 0:
                w = self.L[0] / 2
            elif n == self.nnode - 1:
                w = self.L[n - 1] / 2
            else:
                w = (self.L[n] + self.L[n - 1]) / 2
            mass = cross_section_inertias.A_rho0 * w
            B_Theta_C = cross_section_inertias.B_I_rho0 * w
            self._B_Theta_C.append(B_Theta_C)
            nodalDOF_r_u = self.nodalDOF_r_u[n]
            nodalDOF_p_u = self.nodalDOF_p_u[n]
            self.__M[nodalDOF_r_u, nodalDOF_r_u] = mass * np.eye(3, dtype=float)
            self.__M[nodalDOF_p_u, nodalDOF_p_u] = B_Theta_C
        self._B_Theta_C = np.array(self._B_Theta_C)

        self._sensors = []

        # allocate memery
        self._B_Omega_q = np.zeros((3, 14), dtype=float)
        self._B_J_R = np.zeros((3, 12), dtype=float)
        self._B_J_R_q = np.zeros((3, 12, 14), dtype=float)
        self._B_Psi_q = np.zeros((3, 14), dtype=float)
        self._B_Psi_u = np.zeros((3, 12), dtype=float)
        # CooMatrix
        self._c_q_coo = CooMatrix((self.nla_c, self.nq))
        self._W_c_coo = CooMatrix((self.nu, self.nla_c))
        self._Wla_c_q_coo = CooMatrix((self.nu, self.nq))
        self.__c_la_c = CooMatrix((self.nla_c, self.nla_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            elDOF = np.arange(elDOF.start, elDOF.stop)
            elDOF_u = np.arange(elDOF_u.start, elDOF_u.stop)
            elDOF_la_c = np.arange(elDOF_la_c.start, elDOF_la_c.stop)
            #
            self._c_q_coo.allocate_data(elDOF_la_c, elDOF)
            self._W_c_coo.allocate_data(elDOF_u, elDOF_la_c)
            self._Wla_c_q_coo.allocate_data(elDOF_u, elDOF)
            self.__c_la_c.allocate_data(elDOF_la_c, elDOF_la_c)
        self._c_q_coo.fix_size()
        self._W_c_coo.fix_size()
        self._Wla_c_q_coo.fix_size()
        self.__c_la_c.fix_size()

        self._q_dot_q_coo = CooMatrix((self.nq, self.nq))
        self._q_dot_u_coo = CooMatrix((self.nq, self.nu))
        self._h_u_coo = CooMatrix((self.nu, self.nu))
        self._g_S_q_coo = CooMatrix((self.nla_S, self.nq))
        for n in range(self.nnode):
            nodalDOF_r = self.nodalDOF_r[n]
            nodalDOF_r_u = self.nodalDOF_r_u[n]
            nodalDOF_p = self.nodalDOF_p[n]
            nodalDOF_p_u = self.nodalDOF_p_u[n]
            nodalDOF_r = np.arange(nodalDOF_r.start, nodalDOF_r.stop)
            nodalDOF_r_u = np.arange(nodalDOF_r_u.start, nodalDOF_r_u.stop)
            nodalDOF_p = np.arange(nodalDOF_p.start, nodalDOF_p.stop)
            nodalDOF_p_u = np.arange(nodalDOF_p_u.start, nodalDOF_p_u.stop)

            self._q_dot_q_coo.allocate_data(nodalDOF_p, nodalDOF_p)
            for a, b in zip(nodalDOF_r, nodalDOF_r_u):
                self._q_dot_u_coo.allocate_data([a], [b])
            self._h_u_coo.allocate_data(nodalDOF_p_u, nodalDOF_p_u)
            self._g_S_q_coo.allocate_data([n], nodalDOF_p)
        for n in range(self.nnode):
            nodalDOF_p = self.nodalDOF_p[n]
            nodalDOF_p_u = self.nodalDOF_p_u[n]
            nodalDOF_p = np.arange(nodalDOF_p.start, nodalDOF_p.stop)
            nodalDOF_p_u = np.arange(nodalDOF_p_u.start, nodalDOF_p_u.stop)
            self._q_dot_u_coo.allocate_data(nodalDOF_p, nodalDOF_p_u)
        self._q_dot_q_coo.fix_size()
        self._q_dot_u_coo.fix_size()
        self._h_u_coo.fix_size()
        self._g_S_q_coo.fix_size()
        # constant terms
        for n in range(self.nnode):
            for i in range(3):
                self._q_dot_u_coo.set_allocated_data(3 * n + i, 1.0)

        # cache
        self._alpha_cache = LRUCache(maxsize=self.nnode * 10)
        self._eval_kinematics_cache = LRUCache(maxsize=self.nnode * 10)

    def set_reference_strains(self, Q):
        self.L = np.array(
            [
                norm(Q[self.nodalDOF_r[el + 1]] - Q[self.nodalDOF_r[el]])
                for el in range(self.nelement)
            ]
        )
        self.B_Gamma0 = []
        self.B_Kappa0 = []
        _, self.B_Gamma0, self.B_Kappa0 = self._eval_els(self._view_element_q(Q))

    def element_number(self, xi):
        num = int(xi * self.nelement)
        return num if num < self.nelement else num - 1

    def element_interval(self, el):
        return (self.xis[el], self.xis[el + 1])

    def _view_element_q(self, q):
        stride = q.strides[0]
        return as_strided(q, shape=(self.nelement, 14), strides=(stride * 7, stride))

    def _view_element_la_c(self, la_c):
        return la_c.reshape((self.nelement, _nla_c_el))

    def _view_nodal_q(self, q):
        return q.reshape((self.nnode, 7))

    def _view_nodal_u(self, u):
        return u.reshape((self.nnode, 6))

    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    def get_sensor(self, xi):
        alpha = self._alpha(xi)
        s = Sensor(xi, alpha)
        self._sensors.append(s)
        return s

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        nnode = nelement + 1
        x0 = np.linspace(0, L, num=nnode)
        y0 = np.zeros(nnode)
        z0 = np.zeros(nnode)
        r_OC = np.vstack((x0, y0, z0))
        r_OC = r_OP0 + (A_IB0 @ r_OC).T
        P = np.repeat(Log_SO3_quat(A_IB0)[None, :], nnode, axis=0)
        return np.hstack((r_OC, P)).flatten()

    @staticmethod
    def serret_frenet_configuration(
        nelement,
        r_OP,
        r_OP_xi,
        r_OP_xixi,
        xi1,
        alpha=0.0,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for a pre-curved rod along curve r_OP. The cross-section orientations are based on the Serret-Frenet equations and afterwards rotated by alpha."""
        nnodes_r = nelement + 1

        r_OP, r_OP_xi, r_OP_xixi = check_time_derivatives(r_OP, r_OP_xi, r_OP_xixi)
        alpha, _, _ = check_time_derivatives(alpha, None, None)

        xis = np.linspace(0, xi1, nnodes_r)

        # nodal positions and unit quaternions
        r0 = np.zeros((nnodes_r, 3))
        p0 = np.zeros((nnodes_r, 4))

        for i, xii in enumerate(xis):
            r0[i] = r_OP0 + A_IB0 @ r_OP(xii)
            r_xi = r_OP_xi(xii)
            r_xixi = r_OP_xixi(xii)
            ex = r_xi / norm(r_xi)
            ey = r_xixi - ex * (ex @ r_xixi)
            ey = ey / norm(ey)
            A_B0B = np.vstack([ex, ey, cross3(ex, ey)]).T
            A_IB = A_IB0 @ A_B0B @ A_IB_basic(alpha(xii)).x
            p0[i] = Log_SO3_quat(A_IB)

        # check for the right quaternion hemisphere
        for i in range(nnodes_r - 1):
            inner = p0[i] @ p0[i + 1]
            if inner < 0:
                p0[i + 1] *= -1

        return np.concatenate([r0, p0], axis=1).flatten()

    @staticmethod
    def pose_configuration(
        nelement,
        r_OP,
        A_IB,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for a pre-curved rod with centerline curve r_OP and orientation of A_IB."""
        nnodes_r = nelement + 1

        assert callable(r_OP), "r_OP must be callable!"
        assert callable(A_IB), "A_IB must be callable!"

        xis = np.linspace(0, 1, nnodes_r)

        # nodal positions and unit quaternions
        r0 = np.zeros((nnodes_r, 3))
        p0 = np.zeros((nnodes_r, 4))

        for i, xii in enumerate(xis):
            r0[i] = r_OP0 + A_IB0 @ r_OP(xii)
            A_IBi = A_IB0 @ A_IB(xii)
            p0[i] = Log_SO3_quat(A_IBi)

        # check for the right quaternion hemisphere
        for i in range(nnodes_r - 1):
            inner = p0[i] @ p0[i + 1]
            if inner < 0:
                p0[i + 1] *= -1

        return np.concatenate([r0, p0], axis=1).flatten()

    def assembler_callback(self):
        self._c_la_c_coo()
        for s in self._sensors:
            num = self.element_number(s.xi)
            s.t0 = self.t0
            s.q0 = self.q0[self.elDOF[num]]
            s.qDOF = self.qDOF[self.elDOF[num]]
            s.uDOF = self.qDOF[self.elDOF_u[num]]

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return np.asanyarray(
            _q_dot_nodes(self._view_nodal_q(q), self._view_nodal_u(u))
        ).ravel()

    def q_dot_q(self, t, q, u):
        self._q_dot_q_coo.data[:] = _p_dot_q_nodes(
            self._view_nodal_q(q), self._view_nodal_u(u)
        ).ravel()
        return self._q_dot_q_coo

    def q_dot_u(self, t, q):
        self._q_dot_u_coo.data[-self.nnode * 12 :] = math_jax.T_SO3_inv_quat_batch(
            self._view_nodal_q(q)[:, 3:], False
        ).ravel()

        return self._q_dot_u_coo

    def step_callback(self, t, q, u):
        p = self._view_nodal_q(q)[:, 3:]
        p /= np.linalg.norm(p, axis=1)[:, None]
        return q, u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.__M

    def h(self, t, q, u):
        return np.asarray(_h_nodes(self._view_nodal_u(u), self._B_Theta_C)).ravel()

    def h_u(self, t, q, u):
        self._h_u_coo.data[:] = _h_u_nodes(
            self._view_nodal_u(u)[:, 3:], self._B_Theta_C
        ).ravel()
        return self._h_u_coo

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        p = q.reshape((self.nnode, 7))[:, 3:]
        return np.sum(p**2, axis=1) - 1

    def g_S_q(self, t, q):
        p = q.reshape((self.nnode, 7))[:, 3:]
        self._g_S_q_coo.data[:] = 2 * p.flatten()
        return self._g_S_q_coo

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        _, B_Gamma, B_Kappa = self._eval_els(self._view_element_q(q))
        la_c_el = _la_c_els(
            B_Gamma,
            B_Kappa,
            self.L,
            self.B_Gamma0,
            self.B_Kappa0,
            self.__c_la_c_el_inv,
        )
        return la_c_el.ravel()

    def c(self, t, q, u, la_c):
        _, B_Gamma, B_Kappa = self._eval_els(self._view_element_q(q))
        return np.asarray(
            _c_els(
                B_Gamma,
                B_Kappa,
                self._view_element_la_c(la_c),
                self.L,
                self.B_Gamma0,
                self.B_Kappa0,
                self.C_n_inv,
                self.C_m_inv,
            )
        ).ravel()

    def c_la_c(self):
        return self.__c_la_c

    def _c_la_c_coo(self):
        self.__c_la_c_el_inv = []
        for el in range(self.nelement):
            c_la_c_el = np.zeros((_nla_c_el, _nla_c_el), dtype=float)
            c_la_c_el[:3, :3] = self.C_n_inv
            c_la_c_el[3:6, 3:6] = self.C_m_inv
            if _nla_c_el == 12:
                c_la_c_el[6:9, 6:9] = self.C_n_inv
                c_la_c_el[9:, 9:] = self.C_m_inv
            c_la_c_el *= self.L[el]
            self.__c_la_c.set_allocated_data(el, c_la_c_el)
            self.__c_la_c_el_inv.append(np.linalg.inv(c_la_c_el))
        self.__c_la_c_el_inv = np.array(self.__c_la_c_el_inv)

    def c_q(self, t, q, u, la_c):
        _, B_Gamma_qe, B_Kappa_qe = self._deval_els(self._view_element_q(q))
        self._c_q_coo.data[:] = _c_q_els(B_Gamma_qe, B_Kappa_qe, self.L).ravel()
        return self._c_q_coo

    def W_c(self, t, q):
        A_IB, B_Gamma, B_Kappa = self._eval_els(self._view_element_q(q))
        self._W_c_coo.data[:] = _W_c_els(A_IB, B_Gamma, B_Kappa, self.L).ravel()
        return self._W_c_coo

    def Wla_c_q(self, t, q, la_c):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval_els(self._view_element_q(q))
        self._Wla_c_q_coo.data[:] = _Wla_c_q_els(
            A_IB_qe, B_Gamma_qe, B_Kappa_qe, self._view_element_la_c(la_c), self.L
        ).ravel()
        return self._Wla_c_q_coo

    # @cachedmethod(lambda self: self._alpha_cache, key=lambda self, xi: xi)
    def _alpha(self, xi):
        num = self.element_number(xi)
        return (xi - self.xis[num]) / (self.xis[num + 1] - self.xis[num])

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, xi):
        el = self.element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, xi):
        el = self.element_number(xi)
        return self.elDOF_u[el]

    def local_qDOF_P(self, xi):
        return self.elDOF_P(xi)

    def local_uDOF_P(self, xi=None):
        return self.elDOF_P_u(xi)

    ##########################
    # r_OP / A_IB contribution
    ##########################
    @cachedmethod(
        lambda self: self._eval_kinematics_cache,
        key=lambda self, qe, xi, B_r_CP=np.zeros(3, dtype=float): (
            qe.tobytes(),
            xi,
            B_r_CP.tobytes(),
        ),
    )
    def _element_kinematics(self, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return _eval_kinematics(alpha, qe, B_r_CP)

    def r_OP(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self._element_kinematics(qe, xi, B_r_CP)[0]

    def r_OP_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self._element_kinematics(qe, xi, B_r_CP)[1]

    def v_P(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        alpha = self._alpha(xi)

        # centerline velocity
        v_C0 = ue[:3]
        v_C1 = ue[6:9]
        v_C = v_C0 + alpha * (v_C1 - v_C0)

        if B_r_CP.any():
            A_IB = self.A_IB(t, qe, xi)
            B_Omega = self.B_Omega(t, qe, ue, xi)
            return v_C + A_IB @ cross3(B_Omega, B_r_CP)
        else:
            return v_C

    def v_P_q(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        if B_r_CP.any():
            A_IB_q = self.A_IB_q(t, qe, xi)
            B_Omega = self.B_Omega(t, qe, ue, xi)
            return cross3(B_Omega, B_r_CP) @ A_IB_q
        else:
            return np.zeros((3, 14), dtype=float)

    def J_P(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self._element_kinematics(qe, xi, B_r_CP)[4]

    def J_P_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self._element_kinematics(qe, xi, B_r_CP)[5]

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=np.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        # centerline acceleration
        a_C0 = ue_dot[:3]
        a_C1 = ue_dot[6:9]
        a_C = a_C0 + alpha * (a_C1 - a_C0)
        if B_r_CP.any():
            A_IB = self.A_IB(t, qe, xi)
            B_Omega = self.B_Omega(t, qe, ue, xi)
            B_Psi = self.B_Psi(t, qe, ue, ue_dot, xi)
            # rigid body formular
            return a_C + A_IB @ (
                cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP))
            )
        else:
            return a_C

    def a_P_q(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        raise

    #     B_Omega = self.B_Omega(t, qe, ue, xi)
    #     B_Psi = self.B_Psi(t, qe, ue, ue_dot, xi)
    #     a_P_q = np.einsum(
    #         "ijk,j->ik",
    #         self.A_IB_q(t, qe, xi),
    #         cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP)),
    #     )
    #     return a_P_q

    def a_P_u(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        raise

    #     B_Omega = self.B_Omega(t, qe, ue, xi)
    #     local = -self.A_IB(t, qe, xi) @ (
    #         ax2skew(cross3(B_Omega, B_r_CP)) + ax2skew(B_Omega) @ ax2skew(B_r_CP)
    #     )

    #     N, _ = self.basis_functions_r(xi)
    #     a_P_u = np.zeros((3, self.nu_element), dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         a_P_u[:, self.nodalDOF_element_p_u[node]] += N[node] * local

    #     return a_P_u

    def A_IB(self, t, qe, xi):
        return self._element_kinematics(qe, xi)[2]

    def A_IB_q(self, t, qe, xi):
        return self._element_kinematics(qe, xi)[3]

    def B_Omega(self, t, qe, ue, xi):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        angular velocities in the B-frame.
        """
        alpha = self._alpha(xi)
        B_Omega_1 = ue[3:6]
        B_Omega_2 = ue[9:12]
        B_Omega = B_Omega_1 + alpha * (B_Omega_2 - B_Omega_1)
        return B_Omega

    def B_Omega_q(self, t, qe, ue, xi):
        return self._B_Omega_q

    def B_J_R(self, t, qe, xi):
        alpha = self._alpha(xi)
        np.fill_diagonal(self._B_J_R[:, 3:6], 1 - alpha)
        np.fill_diagonal(self._B_J_R[:, 9:12], alpha)
        return self._B_J_R

    def B_J_R_q(self, t, qe, xi):
        return self._B_J_R_q

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        alpha = self._alpha(xi)
        B_Psi_1 = ue_dot[3:6]
        B_Psi_2 = ue_dot[9:12]
        B_Psi = B_Psi_1 + alpha * (B_Psi_2 - B_Psi_1)
        return B_Psi

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        return self._B_Psi_q

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        return self._B_Psi_u

    def _eval_els(self, q_els):
        return _eval_els(q_els, self.L)

    def _deval_els(self, q_els):
        return _deval_els(q_els, self.L)

    def _eval_deval_els(self, q_els):
        return _eval_deval_els(q_els, self.L)


@njit(cache=True)
def _eval_kinematics(alpha, qe, B_r_CP):
    r_OC0, P0, r_OC1, P1 = np.split(qe, [3, 7, 10])

    r_OP = (1 - alpha) * r_OC0 + alpha * r_OC1
    P = (1 - alpha) * P0 + alpha * P1

    P_qe = np.zeros((4, 14), dtype=float)
    np.fill_diagonal(P_qe[:, 3:7], 1 - alpha)
    np.fill_diagonal(P_qe[:, 10:], alpha)

    A_IB = Exp_SO3_quat(P, normalize=True)
    A_P = Exp_SO3_quat_P(P, normalize=True)
    A_IB_qe = np.empty((3, 3, 14))
    for i in range(3):
        A_IB_qe[i] = A_P[i] @ P_qe

    #
    r_OP_qe = np.zeros((3, 14), dtype=float)
    np.fill_diagonal(r_OP_qe[:, :3], 1 - alpha)
    np.fill_diagonal(r_OP_qe[:, 7:10], alpha)

    J_P = np.zeros((3, 12), dtype=float)
    J_P_q = np.zeros((3, 12, 14), dtype=float)
    np.fill_diagonal(J_P[:, :3], 1 - alpha)
    np.fill_diagonal(J_P[:, 6:9], alpha)
    if B_r_CP.any():
        # r_OP
        r_OP += A_IB @ B_r_CP
        # r_OP_qe
        for i in range(3):
            r_OP_qe[i] += B_r_CP @ A_IB_qe[i]
        # J_P
        B_r_CP_tilde = ax2skew(B_r_CP)
        r_CP_tilde = A_IB @ B_r_CP_tilde
        J_P[:, 3:6] = -(1 - alpha) * r_CP_tilde
        J_P[:, 9:12] = -alpha * r_CP_tilde
        # J_P_q
        r_CP_tilde_q = np.zeros((3, 3, 14), dtype=float)
        for i in range(3):
            r_CP_tilde_q[i] = B_r_CP_tilde.T @ A_IB_qe[i]
        J_P_q[:, 3:6] = -(1 - alpha) * r_CP_tilde_q
        J_P_q[:, 9:12] = -alpha * r_CP_tilde_q
    return r_OP, r_OP_qe, A_IB, A_IB_qe, J_P, J_P_q


def _h_node(u, B_Theta_C):
    B_omega_IB = u[3:]
    tmp = B_Theta_C @ B_omega_IB
    cross = jnp.cross(tmp, B_omega_IB)
    return jnp.pad(cross, (3, 0))


_h_nodes = jit(vmap(_h_node))


def _h_u_node(B_omega_IB, B_Theta_C):
    return (
        math_jax.ax2skew(B_Theta_C @ B_omega_IB)
        - math_jax.ax2skew(B_omega_IB) @ B_Theta_C
    )


_h_u_nodes = jit(vmap(_h_u_node))


def _q_dot_node(q, u):
    T = math_jax.T_SO3_inv_quat(q[3:], normalize=False) @ u[3:]
    return jnp.concatenate([u[:3], T])


_q_dot_nodes = jit(vmap(_q_dot_node))


def _p_dot_p_node(q, u):
    return u[3:] @ math_jax.T_SO3_inv_quat_P(q[3:], normalize=False)


_p_dot_q_nodes = jit(vmap(_p_dot_p_node))


def _la_c_el(
    B_Gamma,
    B_Kappa,
    Le,
    B_Gamma0,
    B_Kappa0,
    c_la_c_el_inv,
):
    eps = jnp.concatenate(
        [
            (B_Gamma - B_Gamma0) * Le,
            (B_Kappa - B_Kappa0) * Le,
        ]
    )
    # TODO: add damping
    return c_la_c_el_inv @ eps


_la_c_els = jit(vmap(_la_c_el))


def _Wla_c_q_el(A_IB_qe, B_Gamma_qe, B_Kappa_qe, la_c, Le):
    B_n = la_c[:3]
    B_m = la_c[3:]

    W0 = B_n @ A_IB_qe

    common = (
        -0.5
        * Le
        * (
            jnp.cross(B_n[:, None], B_Gamma_qe, axis=0)
            + jnp.cross(B_m[:, None], B_Kappa_qe, axis=0)
        )
    )

    return jnp.concatenate([W0, common, -W0, common])


_Wla_c_q_els = jit(vmap(_Wla_c_q_el))


def _c_el(B_Gamma, B_Kappa, la_c, Le, B_Gamma0, B_Kappa0, C_n_inv, C_m_inv):
    B_n, B_m = la_c[:3], la_c[3:]

    c_n = (C_n_inv @ B_n - (B_Gamma - B_Gamma0)) * Le
    c_m = (C_m_inv @ B_m - (B_Kappa - B_Kappa0)) * Le

    # TODO:add damping
    return jnp.concatenate([c_n, c_m])


_c_els = jit(vmap(_c_el, in_axes=(0, 0, 0, 0, 0, 0, None, None)))


def _c_q_el(B_Gamma_qe, B_Kappa_qe, Le):
    c_n_qe = -B_Gamma_qe * Le
    c_m_qe = -B_Kappa_qe * Le
    return jnp.concatenate([c_n_qe, c_m_qe])


_c_q_els = jit(vmap(_c_q_el))


def _W_c_el(A_IB, B_Gamma, B_Kappa, Le):
    # A_IB, B_Gamma, B_Kappa = _eval(qe, Le)
    s1 = 0.5 * Le * math_jax.ax2skew(B_Gamma)
    s2 = 0.5 * Le * math_jax.ax2skew(B_Kappa)

    # TODO:add damping
    row1 = jnp.concatenate([A_IB, zeros3], axis=1)
    row2 = jnp.concatenate([s1, eye3 + s2], axis=1)
    row3 = jnp.concatenate([-A_IB, zeros3], axis=1)
    row4 = jnp.concatenate([s1, -eye3 + s2], axis=1)

    return jnp.concatenate([row1, row2, row3, row4], axis=0)


_W_c_els = jit(vmap(_W_c_el))


def _eval(qe, Le):
    r_OC0 = qe[:3]
    P0 = qe[3:7]
    r_OC1 = qe[7:10]
    P1 = qe[10:14]

    inv_Le = 1.0 / Le

    r_OC_s = (r_OC1 - r_OC0) * inv_Le

    P = 0.5 * (P0 + P1)
    P_s = (P1 - P0) * inv_Le

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    B_Gamma = A_IB.T @ r_OC_s

    B_Kappa = T @ P_s
    return A_IB, B_Gamma, B_Kappa


_eval_els = jit(vmap(_eval))


def _deval(qe, Le):
    r_OC0 = qe[:3]
    P0 = qe[3:7]
    r_OC1 = qe[7:10]
    P1 = qe[10:14]

    inv_Le = 1.0 / Le

    r_OC_s = (r_OC1 - r_OC0) * inv_Le

    P = (P0 + P1) / 2
    P_s = (P1 - P0) * inv_Le
    P_qe = 0.5 * jnp.hstack(
        (jnp.zeros((4, 3)), jnp.eye(4), jnp.zeros((4, 3)), jnp.eye(4))
    )

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    A_IB_T = A_IB.T
    A_IB_qe = math_jax.Exp_SO3_quat_P(P, normalize=True) @ P_qe
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    T_P = math_jax.T_SO3_quat_P(P, normalize=True)

    # B_Gamma = A_IB.T @ r_OC_s
    term2 = (
        jnp.concatenate([-A_IB_T, jnp.zeros((3, 4)), A_IB_T, jnp.zeros((3, 4))], axis=1)
        * inv_Le
    )

    B_Gamma_qe = jnp.einsum("k,kij", r_OC_s, A_IB_qe) + term2

    # B_Kappa = T @ P_s
    term2 = (
        jnp.concatenate([jnp.zeros((3, 3)), -T, jnp.zeros((3, 3)), T], axis=1) * inv_Le
    )
    B_Kappa_qe = P_s @ T_P @ P_qe + term2

    return A_IB_qe, B_Gamma_qe, B_Kappa_qe


_deval_els = jit(vmap(_deval))


def _eval_deval(qe, Le):
    r_OC0 = qe[:3]
    P0 = qe[3:7]
    r_OC1 = qe[7:10]
    P1 = qe[10:14]

    inv_Le = 1.0 / Le

    r_OC_s = (r_OC1 - r_OC0) * inv_Le

    P = (P0 + P1) / 2
    P_s = (P1 - P0) * inv_Le
    P_qe = 0.5 * jnp.hstack(
        (jnp.zeros((4, 3)), jnp.eye(4), jnp.zeros((4, 3)), jnp.eye(4))
    )

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    A_IB_T = A_IB.T
    A_IB_qe = math_jax.Exp_SO3_quat_P(P, normalize=True) @ P_qe
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    T_P = math_jax.T_SO3_quat_P(P, normalize=True)

    B_Gamma = A_IB.T @ r_OC_s
    term2 = (
        jnp.concatenate([-A_IB_T, jnp.zeros((3, 4)), A_IB_T, jnp.zeros((3, 4))], axis=1)
        * inv_Le
    )

    B_Gamma_qe = jnp.einsum("k,kij", r_OC_s, A_IB_qe) + term2

    B_Kappa = T @ P_s
    term2 = (
        jnp.concatenate([jnp.zeros((3, 3)), -T, jnp.zeros((3, 3)), T], axis=1) * inv_Le
    )
    B_Kappa_qe = P_s @ T_P @ P_qe + term2

    return A_IB, B_Gamma, B_Kappa, A_IB_qe, B_Gamma_qe, B_Kappa_qe


_eval_deval_els = jit(vmap(_eval_deval))
