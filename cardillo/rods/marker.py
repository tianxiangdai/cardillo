import numpy as np
from numba import njit

from cardillo.math_numba import cross3, ax2skew, Exp_SO3_quat, Exp_SO3_quat_P
from ..utility.cachetools import MyLRUCache


class Marker:
    def __init__(self, xi, alpha):
        self.xi = xi
        self.alpha = alpha
        self._local_qDOF_P = slice(0, 14)
        self._local_uDOF_P = slice(0, 12)

        # allocate memery
        self._B_Omega_q = np.zeros((3, 14), dtype=float)
        self._B_J_R = np.zeros((3, 12), dtype=float)
        self._B_J_R[0, 3] = self._B_J_R[1, 4] = self._B_J_R[2, 5] = 1 - alpha
        self._B_J_R[0, 9] = self._B_J_R[1, 10] = self._B_J_R[2, 11] = alpha
        self._B_J_R_q = np.zeros((3, 12, 14), dtype=float)
        self._B_Psi_q = np.zeros((3, 14), dtype=float)
        self._B_Psi_u = np.zeros((3, 12), dtype=float)

        self._A_IB_cache = MyLRUCache(maxsize=5)
        self._A_IB_q_cache = MyLRUCache(maxsize=5)

    ####################################################
    # interactions with other bodies and the environment
    ####################################################

    def local_qDOF_P(self, xi=None):
        return self._local_qDOF_P

    def local_uDOF_P(self, xi=None):
        return self._local_uDOF_P

    ##########################
    # r_OP / A_IB contribution
    ##########################

    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q, xi)
        return _r_OP(self.alpha, q, A_IB, B_r_CP)

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        A_IB_q = self.A_IB_q(t, q, xi)
        return _r_OP_q(self.alpha, A_IB_q, B_r_CP)

    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q, xi)
        return _v_P(self.alpha, A_IB, u, self.B_Omega(t, q, u, xi), B_r_CP)

    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        A_IB_q = self.A_IB_q(t, q, xi)
        B_Omega = self.B_Omega(t, q, u, xi)
        return _v_P_q(A_IB_q, B_Omega, B_r_CP)

    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q, xi)
        return _J_P(self.alpha, A_IB, B_r_CP)

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        A_IB_q = self.A_IB_q(t, q, xi)
        return _J_P_q(self.alpha, A_IB_q, B_r_CP)

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        # centerline acceleration
        a_C0 = u_dot[:3]
        a_C1 = u_dot[6:9]
        a_C = a_C0 + self.alpha * (a_C1 - a_C0)
        if B_r_CP.any():
            A_IB = self.A_IB(t, q, xi)
            B_Omega = self.B_Omega(t, q, u, xi)
            B_Psi = self.B_Psi(t, q, u, u_dot, xi)
            # rigid body formular
            return a_C + A_IB @ (
                cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP))
            )
        else:
            return a_C

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=None):
        raise

    #     B_Omega = self.B_Omega(t, q, u, xi)
    #     B_Psi = self.B_Psi(t, q, u, u_dot, xi)
    #     a_P_q = np.einsum(
    #         "ijk,j->ik",
    #         self.A_IB_q(t, q, xi),
    #         cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP)),
    #     )
    #     return a_P_q

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=None):
        raise

    #     B_Omega = self.B_Omega(t, q, u, xi)
    #     local = -self.A_IB(t, q, xi) @ (
    #         ax2skew(cross3(B_Omega, B_r_CP)) + ax2skew(B_Omega) @ ax2skew(B_r_CP)
    #     )

    #     N, _ = self.basis_functions_r(xi)
    #     a_P_u = np.zeros((3, self.nu_element), dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         a_P_u[:, self.nodalDOF_element_p_u[node]] += N[node] * local

    #     return a_P_u

    def A_IB(self, t, q, xi=None):
        key = q.tobytes()
        A_IB = self._A_IB_cache[key]
        if A_IB is None:
            A_IB = _A_IB(self.alpha, q)
            self._A_IB_cache[key] = A_IB
        return A_IB

    # @cachedmethod(lambda self: self._A_IB_q_cache, key=lambda self, t, q, xi: q.tobytes())
    def A_IB_q(self, t, q, xi=None):
        key = q.tobytes()
        A_IB_q = self._A_IB_q_cache[key]
        if A_IB_q is None:
            A_IB_q = _A_IB_q(self.alpha, q)
            self._A_IB_q_cache[key] = A_IB_q
        return A_IB_q

    def B_Omega(self, t, q, u, xi=None):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        angular velocities in the B-frame.
        """
        return _B_Omega(self.alpha, u)

    def B_Omega_q(self, t, q, u, xi=None):
        return self._B_Omega_q

    def B_J_R(self, t, q, xi=None):
        return self._B_J_R

    def B_J_R_q(self, t, q, xi=None):
        return self._B_J_R_q

    def B_Psi(self, t, q, u, u_dot, xi=None):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        B_Psi_1 = u_dot[3:6]
        B_Psi_2 = u_dot[9:12]
        B_Psi = B_Psi_1 + self.alpha * (B_Psi_2 - B_Psi_1)
        return B_Psi

    def B_Psi_q(self, t, q, u, u_dot, xi=None):
        return self._B_Psi_q

    def B_Psi_u(self, t, q, u, u_dot, xi=None):
        return self._B_Psi_u


@njit(cache=True)
def _r_OP(alpha, q, A_IB, B_r_CP):
    r_OC0, r_OC1 = q[:3], q[7:10]
    r_OP = (1 - alpha) * r_OC0 + alpha * r_OC1
    if B_r_CP.any():
        r_OP += A_IB @ B_r_CP
    return r_OP


@njit(cache=True)
def _r_OP_q(alpha, A_IB_q, B_r_CP):
    r_OP_q = np.zeros((3, 14), dtype=float)
    r_OP_q[0, 0] = r_OP_q[1, 1] = r_OP_q[2, 2] = 1 - alpha
    r_OP_q[0, 7] = r_OP_q[1, 8] = r_OP_q[2, 9] = alpha
    if B_r_CP.any():
        for i in range(3):
            r_OP_q[i] += B_r_CP @ A_IB_q[i]
    return r_OP_q


@njit(cache=True)
def _v_P(alpha, A_IB, u, B_Omega, B_r_CP):
    v_C0 = u[:3]
    v_C1 = u[6:9]
    v_C = v_C0 + alpha * (v_C1 - v_C0)

    if B_r_CP.any():
        return v_C + A_IB @ cross3(B_Omega, B_r_CP)
    else:
        return v_C


@njit(cache=True)
def _v_P_q(A_IB_q, B_Omega, B_r_CP):
    v_P_q = np.zeros((3, 14), dtype=float)
    if B_r_CP.any():
        cross = cross3(B_Omega, B_r_CP)
        for i in range(3):
            v_P_q[i] = cross @ A_IB_q[i]
    return v_P_q


@njit(cache=True)
def _J_P(alpha, A_IB, B_r_CP):
    J_P = np.zeros((3, 12), dtype=float)
    J_P[0, 0] = J_P[1, 1] = J_P[2, 2] = 1 - alpha
    J_P[0, 6] = J_P[1, 7] = J_P[2, 8] = alpha
    if B_r_CP.any():
        B_r_CP_tilde = ax2skew(B_r_CP)
        r_CP_tilde = A_IB @ B_r_CP_tilde
        J_P[:, 3:6] = -(1 - alpha) * r_CP_tilde
        J_P[:, 9:12] = -alpha * r_CP_tilde
    return J_P


@njit(cache=True)
def _J_P_q(alpha, A_IB_q, B_r_CP):
    J_P_q = np.zeros((3, 12, 14), dtype=float)
    if B_r_CP.any():
        B_r_CP_tilde = ax2skew(B_r_CP)
        r_CP_tilde_q = np.zeros((3, 3, 14), dtype=float)
        for i in range(3):
            r_CP_tilde_q[i] = B_r_CP_tilde.T @ A_IB_q[i]
        J_P_q[:, 3:6] = -(1 - alpha) * r_CP_tilde_q
        J_P_q[:, 9:12] = -alpha * r_CP_tilde_q
    return J_P_q


@njit(cache=True)
def _A_IB(alpha, q):
    P0, P1 = q[3:7], q[10:]
    P = (1 - alpha) * P0 + alpha * P1
    return Exp_SO3_quat(P, normalize=True)


@njit(cache=True)
def _A_IB_q(alpha, q):
    P0, P1 = q[3:7], q[10:]
    P = (1 - alpha) * P0 + alpha * P1

    P_q = np.zeros((4, 14), dtype=float)

    P_q[0, 3] = P_q[1, 4] = P_q[2, 5] = P_q[3, 6] = 1 - alpha
    P_q[0, 10] = P_q[1, 11] = P_q[2, 12] = P_q[3, 13] = alpha

    A_P = Exp_SO3_quat_P(P, normalize=True)
    A_IB_q = np.empty((3, 3, 14))
    for i in range(3):
        A_IB_q[i] = A_P[i] @ P_q
    return A_IB_q


@njit(cache=True)
def _B_Omega(alpha, u):
    B_Omega_1 = u[3:6]
    B_Omega_2 = u[9:12]
    return B_Omega_1 + alpha * (B_Omega_2 - B_Omega_1)
