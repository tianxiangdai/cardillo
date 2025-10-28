from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np

from ..math import (
    cross3,
    ax2skew,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)


class RigidBodyKinematics:
    def __init__(self):
        self.A_IB_cache = LRUCache(maxsize=1)
        self.A_IB_q_cache = LRUCache(maxsize=1)
        self.r_OP_cache = LRUCache(maxsize=1)
        self.v_P_cache = LRUCache(maxsize=1)
        self.J_P_cache = LRUCache(maxsize=1)
        self._nq = 7
        self._nu = 6

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self._nq, dtype=np.common_type(q, u))
        q_dot[:3] = u[:3]
        q_dot[3:] = T_SO3_inv_quat(q[3:], normalize=False) @ u[3:]
        return q_dot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self._nq, self._nq), dtype=np.common_type(q, u))
        q_dot_q[3:, 3:] = np.einsum(
            "ijk,j->ik", T_SO3_inv_quat_P(q[3:], normalize=False), u[3:]
        )
        return q_dot_q

    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self._nq, self._nu), dtype=q.dtype)
        q_dot_u[:3, :3] = np.eye(3, dtype=q.dtype)
        q_dot_u[3:, 3:] = T_SO3_inv_quat(q[3:], normalize=False)
        return q_dot_u

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        P = q[3:]
        return np.array([P @ P - 1.0], dtype=q.dtype)

    def g_S_q(self, t, q):
        P = q[3:]
        g_S_q = np.zeros((1, 7), dtype=q.dtype)
        g_S_q[0, 3:] = 2.0 * P
        return g_S_q

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return np.arange(self._nq)

    def local_uDOF_P(self, xi=None):
        return np.arange(self._nu)

    @cachedmethod(
        lambda self: self.A_IB_cache,
        key=lambda self, t, q, xi=None: hashkey(t, *q),
    )
    def A_IB(self, t, q, xi=None):
        return Exp_SO3_quat(q[3:])

    @cachedmethod(
        lambda self: self.A_IB_q_cache,
        key=lambda self, t, q, xi=None: hashkey(t, *q),
    )
    def A_IB_q(self, t, q, xi=None):
        A_IB_q = np.zeros((3, 3, self._nq), dtype=q.dtype)
        A_IB_q[:, :, 3:] = Exp_SO3_quat_p(q[3:])
        return A_IB_q

    @cachedmethod(
        lambda self: self.r_OP_cache,
        key=lambda self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float): hashkey(
            t, *q, *B_r_CP
        ),
    )
    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IB(t, q) @ B_r_CP

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        r_OP_q = np.zeros((3, self._nq), dtype=q.dtype)
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IB_q(t, q), B_r_CP)
        return r_OP_q

    @cachedmethod(
        lambda self: self.v_P_cache,
        key=lambda self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float): hashkey(
            t, *q, *u, *B_r_CP
        ),
    )
    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return u[:3] + self.A_IB(t, q) @ cross3(u[3:], B_r_CP)

    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.einsum("ijk,j->ik", self.A_IB_q(t, q), cross3(u[3:], B_r_CP))

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return u_dot[:3] + self.A_IB(t, q) @ (
            cross3(u_dot[3:], B_r_CP) + cross3(u[3:], cross3(u[3:], B_r_CP))
        )

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.einsum(
            "ijk,j->ik",
            self.A_IB_q(t, q),
            cross3(u_dot[3:], B_r_CP) + cross3(u[3:], cross3(u[3:], B_r_CP)),
        )

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        a_P_u = np.zeros((3, self._nu), dtype=float)
        a_P_u[:, 3:] = -self.A_IB(t, q) @ (
            ax2skew(cross3(u[3:], B_r_CP)) + ax2skew(u[3:]) @ ax2skew(B_r_CP)
        )
        return a_P_u

    @cachedmethod(
        lambda self: self.J_P_cache,
        key=lambda self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float): hashkey(
            t, *q, *B_r_CP
        ),
    )
    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        J_P = np.zeros((3, self._nu), dtype=q.dtype)
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IB(t, q) @ ax2skew(B_r_CP)
        return J_P

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        J_P_q = np.zeros((3, self._nu, self._nq), dtype=q.dtype)
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IB_q(t, q), -ax2skew(B_r_CP))
        return J_P_q

    def kappa_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.A_IB(t, q) @ (cross3(u[3:], cross3(u[3:], B_r_CP)))

    def kappa_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IB_q(t, q), cross3(u[3:], cross3(u[3:], B_r_CP))
        )

    def kappa_P_u(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self._nu))
        kappa_P_u[:, 3:] = -self.A_IB(t, q) @ (
            ax2skew(cross3(u[3:], B_r_CP)) + ax2skew(u[3:]) @ ax2skew(B_r_CP)
        )
        return kappa_P_u

    def B_Omega(self, t, q, u, xi=None):
        return u[3:]

    def B_Omega_q(self, t, q, u, xi=None):
        return np.zeros((3, self._nq), dtype=np.common_type(q, u))

    def B_Psi(self, t, q, u, u_dot, xi=None):
        return u_dot[3:]

    def B_Psi_q(self, t, q, u, u_dot, xi=None):
        return np.zeros((3, self._nq), dtype=np.common_type(q, u, u_dot))

    def B_Psi_u(self, t, q, u, u_dot, xi=None):
        return np.zeros((3, self._nu), dtype=np.common_type(q, u, u_dot))

    def B_kappa_R(self, t, q, u, xi=None):
        return np.zeros(3, dtype=np.common_type(q, u))

    def B_kappa_R_q(self, t, q, u, xi=None):
        return np.zeros((3, self._nq), dtype=np.common_type(q, u))

    def B_kappa_R_u(self, t, q, u, xi=None):
        return np.zeros((3, self._nu), dtype=np.common_type(q, u))

    def B_J_R(self, t, q, xi=None):
        B_J_R = np.zeros((3, self._nu), dtype=q.dtype)
        B_J_R[:, 3:] = np.eye(3)
        return B_J_R

    def B_J_R_q(self, t, q, xi=None):
        return np.zeros((3, self._nu, self._nq), dtype=q.dtype)
