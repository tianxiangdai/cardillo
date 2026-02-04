import numpy as np
from numpy.lib.stride_tricks import as_strided

import jax
from jax import vmap, jit
from jax import numpy as jnp
from numba import njit

from cachetools import cachedmethod, LRUCache

from cardillo.math_numba import (
    norm,
    Log_SO3_quat,
    Exp_SO3_quat,
    T_SO3_inv_quat_P,
    T_SO3_inv_quat,
    cross3,
    ax2skew,
    Exp_SO3_quat_P,
)
from cardillo import math_jax
from cardillo.rods._base import RodExportBase
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.rods import CrossSectionInertias

from cardillo.math import A_IB_basic
from cardillo.utility.check_time_derivatives import check_time_derivatives

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

        # allocate memery
        self._q = self._u = self._la_c = np.empty(0)
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
            self._c_q_coo.allocate(elDOF_la_c, elDOF)
            self._W_c_coo.allocate(elDOF_u, elDOF_la_c)
            self._Wla_c_q_coo.allocate(elDOF_u, elDOF)
            self.__c_la_c.allocate(elDOF_la_c, elDOF_la_c)
        self._c_q_coo.fix_size()
        self._W_c_coo.fix_size()
        self._Wla_c_q_coo.fix_size()
        self.__c_la_c.fix_size()

        self._q_dot_q_coo = CooMatrix((self.nq, self.nq))
        self._q_dot_u_coo = CooMatrix((self.nq, self.nq))
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

            self._q_dot_q_coo.allocate(nodalDOF_p, nodalDOF_p)
            self._q_dot_u_coo.allocate(nodalDOF_r, nodalDOF_r_u)
            self._q_dot_u_coo.allocate(nodalDOF_p, nodalDOF_p_u)
            self._h_u_coo.allocate(nodalDOF_p_u, nodalDOF_p_u)
            self._g_S_q_coo.allocate([n], nodalDOF_p)
        self._q_dot_q_coo.fix_size()
        self._q_dot_u_coo.fix_size()
        self._h_u_coo.fix_size()
        self._g_S_q_coo.fix_size()
        # constant terms
        for n in range(self.nnode):
            self._q_dot_u_coo.set_allocated(2 * n, np.eye(3, dtype=float))

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
        A_IB0, B_Gamma0, B_Kappa0 = _eval_batch(self._view_element_q(Q), self.L)
        A_IB0 = np.asarray(A_IB0)
        self.B_Gamma0 = np.asarray(B_Gamma0)
        self.B_Kappa0 = np.asarray(B_Kappa0)

    def element_number(self, xi):
        num = int(xi * self.nelement)
        return num if num < self.nelement else num - 1

    def element_interval(self, el):
        return (self.xis[el], self.xis[el + 1])

    def _view_element_q(self, q):
        stride = q.strides[0]
        return as_strided(q, shape=(self.nelement, 14), strides=(stride * 7, stride))

    def _view_element_la_c(self, la_c):
        stride = la_c.strides[0]
        return as_strided(
            la_c, shape=(self.nelement, _nla_c_el), strides=(stride * _nla_c_el, stride)
        )

    def _view_nodal_q(self, q):
        stride = q.strides[0]
        return as_strided(q, shape=(self.nnode, 7), strides=(stride * 7, stride))

    def _view_nodal_u(self, u):
        stride = u.strides[0]
        return as_strided(u, shape=(self.nnode, 6), strides=(stride * 6, stride))

    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

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

    def update(self, keys, t=None, q=None, u=None, la_c=None, **kwargs):
        q_els, la_c_els = self._view_element_q(q), self._view_element_la_c(la_c)
        q_nodes, u_nodes = self._view_nodal_q(q), self._view_nodal_u(u)
        # W_c
        if "W_c" in keys:
            self._W_c_coo.data[:] = np.asarray(_W_c_el_batch(q_els, self.L)).ravel()
        if "Wla_c_q" in keys:
            self._Wla_c_q_coo.data[:] = np.asarray(
                _Wla_c_el_qe_batch(q_els, la_c_els, self.L)
            ).ravel()
        # c
        if "c" in keys:
            self._c = np.asarray(
                _c_el_batch(
                    q_els,
                    la_c_els,
                    self.L,
                    self.B_Gamma0,
                    self.B_Kappa0,
                    self.C_n_inv,
                    self.C_m_inv,
                )
            ).ravel()
        if "c_q" in keys:
            c_el_qes = _c_el_qe_batch(self._view_element_q(q), self.L)
            self._c_q_coo.data[:] = np.asarray(c_el_qes).ravel()
        # h
        if "h" in keys:
            self._h = np.asarray(_h_node_batch(u_nodes, self._B_Theta_C)).ravel()
        if "h_u" in keys:
            for n in range(self.nnode):
                B_omega_IB = u_nodes[n, 3:]
                self._h_u_coo.set_allocated(
                    n,
                    ax2skew(self._B_Theta_C[n] @ B_omega_IB)
                    - ax2skew(B_omega_IB) @ self._B_Theta_C[n],
                )
        # q_dot
        if "q_dot" in keys:
            self._q_dot = np.asarray(_q_dot_node_batch(q_nodes, u_nodes)).ravel()
        if "q_dot_q" in keys:
            self._q_dot_q_coo.data[:] = np.asarray(
                _p_dot_p_node_batch(q_nodes, u_nodes)
            ).ravel()
        if "q_dot_u" in keys:
            for n in range(self.nnode):
                p = q_nodes[n, 3:]
                self._q_dot_u_coo.set_allocated(
                    2 * n + 1, T_SO3_inv_quat(p, normalize=False)
                )

        #
        self._q = q
        self._u = u
        self._la_c = la_c

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        if self._q.tobytes() != q.tobytes() or self._u.tobytes() != u.tobytes():
            q, u = self._view_nodal_q(q), self._view_nodal_u(u)
            q_dot = _q_dot_node_batch(q, u)
            self._q_dot = np.asarray(q_dot).ravel()
        return self._q_dot

    def q_dot_q(self, t, q, u):
        if self._q.tobytes() != q.tobytes() or self._u.tobytes() != u.tobytes():
            q, u = self._view_nodal_q(q), self._view_nodal_u(u)
            self._q_dot_q_coo.data[:] = np.asarray(_p_dot_p_node_batch(q, u)).ravel()
        return self._q_dot_q_coo

    def q_dot_u(self, t, q):
        if self._q.tobytes() != q.tobytes():
            q = self._view_nodal_q(q)
            for n in range(self.nnode):
                p = q[n, 3:]
                self._q_dot_u_coo.set_allocated(
                    2 * n + 1, T_SO3_inv_quat(p, normalize=False)
                )
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
        if self._u.tobytes() != u.tobytes():
            u = self._view_nodal_u(u)
            h = _h_node_batch(u, self._B_Theta_C)
            self._h = np.asarray(h).ravel()
        return self._h

    def h_u(self, t, q, u):
        if self._u.tobytes() != u.tobytes():
            for n in range(self.nnode):
                nodalDOF_p_u = self.nodalDOF_p_u[n]
                B_omega_IB = u[nodalDOF_p_u]
                self._h_u_coo.set_allocated(
                    n,
                    ax2skew(self._B_Theta_C[n] @ B_omega_IB)
                    - ax2skew(B_omega_IB) @ self._B_Theta_C[n],
                )
        return self._h_u_coo

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        p = q.reshape((self.nnode, 7))[:, 3:]
        return np.array([pi @ pi - 1.0 for pi in p], dtype=float)

    def g_S_q(self, t, q):
        p = q.reshape((self.nnode, 7))[:, 3:]
        for n in range(self.nnode):
            self._g_S_q_coo.set_allocated(n, 2 * p[n])
        return self._g_S_q_coo

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        la_c_el = _la_c_el_batch(
            self._view_element_q(q),
            self.L,
            self.B_Gamma0,
            self.B_Kappa0,
            self.__c_la_c_el_inv,
        )
        return np.asarray(la_c_el).ravel()

    def c(self, t, q, u, la_c):
        if self._la_c.tobytes() != la_c.tobytes() or self._q.tobytes() != q.tobytes():
            _c_els = _c_el_batch(
                self._view_element_q(q),
                self._view_element_la_c(la_c),
                self.L,
                self.B_Gamma0,
                self.B_Kappa0,
                self.C_n_inv,
                self.C_m_inv,
            )
            self._c = np.asarray(_c_els).ravel()
        return self._c

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
            self.__c_la_c.set_allocated(el, c_la_c_el)
            self.__c_la_c_el_inv.append(np.linalg.inv(c_la_c_el))
        self.__c_la_c_el_inv = np.array(self.__c_la_c_el_inv)

    def c_q(self, t, q, u, la_c):
        if self._q.tobytes() != q.tobytes():
            c_el_qes = _c_el_qe_batch(self._view_element_q(q), self.L)
            self._c_q_coo.data[:] = np.asarray(c_el_qes).ravel()

        return self._c_q_coo

    def W_c(self, t, q):
        if self._q.tobytes() != q.tobytes():
            _W_c_els = _W_c_el_batch(self._view_element_q(q), self.L)
            self._W_c_coo.data[:] = np.asarray(_W_c_els).ravel()
        return self._W_c_coo

    def Wla_c_q(self, t, q, la_c):
        if self._q.tobytes() != q.tobytes() or self._la_c.tobytes() != la_c.tobytes():
            W = _Wla_c_el_qe_batch(
                self._view_element_q(q), self._view_element_la_c(la_c), self.L
            )
            self._Wla_c_q_coo.data[:] = np.asarray(W).ravel()
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
        elDOF = self.elDOF[el]
        return np.arange(elDOF.start, elDOF.stop)

    def elDOF_P_u(self, xi):
        num = self.element_number(xi)
        elDOF_u = self.elDOF_u[num]
        return np.arange(elDOF_u.start, elDOF_u.stop)

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
    cross = math_jax.cross3(B_Theta_C @ B_omega_IB, B_omega_IB)
    return jnp.array([0.0, 0.0, 0.0, cross[0], cross[1], cross[2]], dtype=jnp.float64)


_h_node_batch = jit(vmap(_h_node))


def _q_dot_node(q, u):
    T = math_jax.T_SO3_inv_quat(q[3:], normalize=False) @ u[3:]
    return jnp.array([u[0], u[1], u[2], T[0], T[1], T[2], T[3]], dtype=jnp.float64)


_q_dot_node_batch = jit(vmap(_q_dot_node))


def _p_dot_p_node(q, u):
    return u[3:] @ math_jax.T_SO3_inv_quat_P(q[3:], normalize=False)


_p_dot_p_node_batch = jit(vmap(_p_dot_p_node))


def _la_c_el(
    qe,
    Le,
    B_Gamma0,
    B_Kappa0,
    c_la_c_el_inv,
):
    _, B_Gamma, B_Kappa = _eval(qe, Le)
    eps = jnp.concatenate(
        [
            (B_Gamma - B_Gamma0) * Le,
            (B_Kappa - B_Kappa0) * Le,
        ]
    )
    # TODO: add damping
    return c_la_c_el_inv @ eps


_la_c_el_batch = jit(vmap(_la_c_el))


def _Wla_c_el_qe(qe, la_c, Le):
    A_IB_qe, B_Gamma_qe, B_Kappa_qe = _deval(qe, Le)
    B_n, B_m = jnp.split(la_c, [3])

    W0 = B_n @ A_IB_qe

    common = (
        -0.5
        * Le
        * (math_jax.cross3(B_n, B_Gamma_qe) + math_jax.cross3(B_m, B_Kappa_qe))
    )

    W = jnp.vstack([W0, common, -W0, common])
    return W


_Wla_c_el_qe_batch = jit(vmap(_Wla_c_el_qe))


def _c_el(qe, la_c, Le, B_Gamma0, B_Kappa0, C_n_inv, C_m_inv):
    _, B_Gamma, B_Kappa = _eval(qe, Le)
    #
    B_n, B_m = jnp.split(la_c, [3])

    c_n = (C_n_inv @ B_n - (B_Gamma - B_Gamma0)) * Le
    c_m = (C_m_inv @ B_m - (B_Kappa - B_Kappa0)) * Le

    # TODO:add damping
    return jnp.concatenate([c_n, c_m], dtype=jnp.float64)


_c_el_batch = jit(vmap(_c_el, in_axes=(0, 0, 0, 0, 0, None, None)))


def _c_el_qe(qe, Le):
    _, B_Gamma_qe, B_Kappa_qe = _deval(qe, Le)
    c_n_qe = -B_Gamma_qe * Le
    c_m_qe = -B_Kappa_qe * Le
    return jnp.concatenate([c_n_qe, c_m_qe], dtype=jnp.float64)


_c_el_qe_batch = jit(vmap(_c_el_qe))


def _W_c_el(qe, Le):
    A_IB, B_Gamma, B_Kappa = _eval(qe, Le)
    s1 = 0.5 * math_jax.ax2skew(B_Gamma) * Le
    s2 = 0.5 * math_jax.ax2skew(B_Kappa) * Le

    W_c_el = jnp.block(
        [
            [A_IB, zeros3],
            [s1, eye3 + s2],
            [-A_IB, zeros3],
            [s1, -eye3 + s2],
        ]
    )
    # TODO:add damping
    return W_c_el


_W_c_el_batch = jit(vmap(_W_c_el))


def _eval(qe, Le):
    r_OC0, P0, r_OC1, P1 = jnp.split(qe, [3, 7, 10])

    r_OC_s = (r_OC1 - r_OC0) / Le

    P = (P0 + P1) / 2
    P_s = (P1 - P0) / Le

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    B_Gamma = A_IB.T @ r_OC_s

    B_Kappa = T @ P_s
    return A_IB, B_Gamma, B_Kappa


_eval_batch = jit(vmap(_eval))


def _deval(qe, Le):
    r_OC0, P0, r_OC1, P1 = jnp.split(qe, [3, 7, 10])

    r_OC_s = (r_OC1 - r_OC0) / Le
    r_OC_s_qe = (
        jnp.hstack((-jnp.eye(3), jnp.zeros((3, 4)), jnp.eye(3), jnp.zeros((3, 4)))) / Le
    )

    P = (P0 + P1) / 2
    P_s = (P1 - P0) / Le
    P_qe = (
        jnp.hstack((jnp.zeros((4, 3)), jnp.eye(4), jnp.zeros((4, 3)), jnp.eye(4))) / 2
    )
    P_s_qe = (
        jnp.hstack((jnp.zeros((4, 3)), -jnp.eye(4), jnp.zeros((4, 3)), jnp.eye(4))) / Le
    )

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    A_IB_qe = math_jax.Exp_SO3_quat_P(P, normalize=True) @ P_qe
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    # B_Gamma = A_IB.T @ r_OC_s
    B_Gamma_qe = (A_IB_qe.T @ r_OC_s).T + A_IB.T @ r_OC_s_qe
    # B_Gamma_qe = jnp.einsum("k,kij", r_OC_s, A_IB_qe) + A_IB.T @ r_OC_s_qe

    # B_Kappa = T @ P_s
    B_Kappa_qe = P_s @ math_jax.T_SO3_quat_P(P, normalize=True) @ P_qe + T @ P_s_qe
    # return A_IB, B_Gamma, B_Kappa, r_OC_s_qe, A_IB_qe, B_Gamma_qe, B_Kappa_qe
    return A_IB_qe, B_Gamma_qe, B_Kappa_qe


# _deval_batch = jit(vmap(_deval_jax))
