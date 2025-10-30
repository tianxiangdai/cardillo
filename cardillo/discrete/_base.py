import numpy as np

from ..math import (
    cross3,
    ax2skew,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)

eye3 = np.eye(3, dtype=np.float64)

class PositionKinematics:
    def __init__(self):
        self._nq = 3
        self._nu = 3

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.eye(self._nq)

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return np.arange(self._nq)

    def local_uDOF_P(self, xi=None):
        return np.arange(self._nu)

    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        r = np.zeros(3, dtype=q.dtype)
        r[: self._nq] = q
        return r + B_r_CP  # A_IB = np.eye(3)

    def r_OP_q(self, t, q, xi=None, B_r_CP=None):
        return np.eye(3, self._nq)

    def J_P(self, t, q, xi=None, B_r_CP=None):
        return np.eye(3, self._nu, dtype=q.dtype)

    def J_P_q(self, t, q, xi=None, B_r_CP=None):
        return np.zeros((3, self._nu, self._nq))

    def v_P(self, t, q, u, xi=None, B_r_CP=None):
        v_P = np.zeros(3, dtype=np.common_type(q, u))
        v_P[: self._nq] = u
        return v_P

    def v_P_q(self, t, q, u, xi=None, B_r_CP=None):
        return np.zeros((3, self._nq))

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=None):
        a_P = np.zeros(3, dtype=np.common_type(q, u, u_dot))
        a_P[: self._nq] = u_dot
        return a_P

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=None):
        return np.zeros((3, self._nq), dtype=np.common_type(q, u, u_dot))

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=None):
        return np.zeros((3, self._nu), dtype=np.common_type(q, u, u_dot))
    

class PoseKinematics:
    def __init__(self):
        self._nq = 7
        self._nu = 6
        self.__q_dot = np.zeros(self._nq, dtype=np.float64)
        self.__q_dot_u = np.zeros((self._nq, self._nu), dtype=np.float64)
        self.__A_IB_q = np.zeros((3, 3, self._nq), dtype=np.float64)
        self.__r_OP_q = np.zeros((3, self._nq), dtype=np.float64)
        self.__a_P_u = np.zeros((3, self._nu), dtype=np.float64)
        self.__J_P = np.zeros((3, self._nu), dtype=np.float64)
        self.__J_P[:, :3] = eye3
        self.__J_P_q = np.zeros((3, self._nu, self._nq), dtype=np.float64)
        self.__kappa_P_u = np.zeros((3, self._nu))
        self.__B_Omega_q = np.zeros((3, self._nq), dtype=np.float64)
        self.__B_Psi_q = np.zeros((3, self._nq), dtype=np.float64)
        self.__B_Psi_u = np.zeros((3, self._nu), dtype=np.float64)
        self.__B_kappa_R = np.zeros(3, dtype=np.float64)
        self.__B_kappa_R_q = np.zeros((3, self._nq), dtype=np.float64)
        self.__B_kappa_R_u = np.zeros((3, self._nu), dtype=np.float64)
        self.__B_J_R = np.zeros((3, self._nu), dtype=np.float64)
        self.__B_J_R_q = np.zeros((3, self._nu, self._nq), dtype=np.float64)
        self.__local_qDOF_P = np.arange(self._nq)
        self.__local_uDOF_P = np.arange(self._nu)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = self.__q_dot
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
        q_dot_u = self.__q_dot_u
        q_dot_u[:3, :3] = eye3
        q_dot_u[3:, 3:] = T_SO3_inv_quat(q[3:], normalize=False)
        return q_dot_u

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return self.__local_qDOF_P

    def local_uDOF_P(self, xi=None):
        return self.__local_uDOF_P

    def A_IB(self, t, q, xi=None):
        return Exp_SO3_quat(q[3:])

    def A_IB_q(self, t, q, xi=None):
        A_IB_q = self.__A_IB_q
        A_IB_q[:, :, 3:] = Exp_SO3_quat_p(q[3:])
        return A_IB_q

    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IB(t, q) @ B_r_CP

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        r_OP_q = self.__r_OP_q
        r_OP_q[:] = 0
        r_OP_q[:, :3] = eye3
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IB_q(t, q), B_r_CP)
        return r_OP_q

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
        a_P_u = self.__a_P_u
        a_P_u[:, 3:] = -self.A_IB(t, q) @ (
            ax2skew(cross3(u[3:], B_r_CP)) + ax2skew(u[3:]) @ ax2skew(B_r_CP)
        )
        return a_P_u

    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        J_P = self.__J_P
        J_P[:, 3:] = -self.A_IB(t, q) @ ax2skew(B_r_CP)
        return J_P

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        self.__J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IB_q(t, q), -ax2skew(B_r_CP))
        return self.__J_P_q

    def kappa_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.A_IB(t, q) @ (cross3(u[3:], cross3(u[3:], B_r_CP)))

    def kappa_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IB_q(t, q), cross3(u[3:], cross3(u[3:], B_r_CP))
        )

    def kappa_P_u(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        kappa_P_u = self.__kappa_P_u
        kappa_P_u[:, 3:] = -self.A_IB(t, q) @ (
            ax2skew(cross3(u[3:], B_r_CP)) + ax2skew(u[3:]) @ ax2skew(B_r_CP)
        )
        return kappa_P_u

    def B_Omega(self, t, q, u, xi=None):
        return u[3:]

    def B_Omega_q(self, t, q, u, xi=None):
        return self.__B_Omega_q

    def B_Psi(self, t, q, u, u_dot, xi=None):
        return u_dot[3:]

    def B_Psi_q(self, t, q, u, u_dot, xi=None):
        return self.__B_Psi_q

    def B_Psi_u(self, t, q, u, u_dot, xi=None):
        return self.__B_Psi_u

    def B_kappa_R(self, t, q, u, xi=None):
        return self.__B_kappa_R

    def B_kappa_R_q(self, t, q, u, xi=None):
        return self.__B_kappa_R_q

    def B_kappa_R_u(self, t, q, u, xi=None):
        return self.__B_kappa_R_u

    def B_J_R(self, t, q, xi=None):
        B_J_R = self.__B_J_R
        B_J_R[:, 3:] = eye3
        return B_J_R

    def B_J_R_q(self, t, q, xi=None):
        return self.__B_J_R_q
