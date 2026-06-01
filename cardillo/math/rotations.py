import numpy as np
from cardillo.math.algebra import (
    norm,
    cross3,
    ax2skew,
    ax2skew_a,
    LeviCivita3,
    ax2skew_squared,
    outer3,
)

from numba import njit

# for small angles we use first order approximations of the equations since
# most of the SO(3) and SE(3) equations get singular for psi -> 0.
# angle_singular = 1.0e-6
angle_singular = 0.0

eye3 = np.eye(3, dtype=np.float64)


class A_IB_basic:
    """Basic rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = np.sin(phi)
        self.cp = np.cos(phi)

    @property
    def x(self) -> np.ndarray:
        """Rotation around x-axis."""
        # fmt: off
        return np.array([[1,       0,        0],
                         [0, self.cp, -self.sp],
                         [0, self.sp,  self.cp]])
        # fmt: on

    @property
    def y(self) -> np.ndarray:
        """Rotation around y-axis."""
        # fmt: off
        return np.array([[ self.cp, 0, self.sp],
                         [       0, 1,       0],
                         [-self.sp, 0, self.cp]])
        # fmt: on

    @property
    def z(self) -> np.ndarray:
        """Rotation around z-axis."""
        # fmt: off
        return np.array([[self.cp, -self.sp, 0],
                         [self.sp,  self.cp, 0],
                         [      0,        0, 1]])


@njit(cache=True)
def Log_SO3_quat(R: np.ndarray) -> np.ndarray:
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    decision = np.zeros(4, dtype=np.float64)
    decision[:3] = np.diag(R)
    decision[3] = np.trace(R)
    i = np.argmax(decision)

    quat = np.zeros(4, dtype=np.float64)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = np.sqrt(0.5 * R[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (R[k, j] - R[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (R[j, i] + R[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (R[k, i] + R[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * np.sqrt(1 + decision[3])
        quat[1] = (R[2, 1] - R[1, 2]) / (4 * quat[0])
        quat[2] = (R[0, 2] - R[2, 0]) / (4 * quat[0])
        quat[3] = (R[1, 0] - R[0, 1]) / (4 * quat[0])

    return quat


@njit(cache=True)
def Exp_SO3_quat(P, normalize=True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.163), Nuetzi2016 (3.31) and Rucker2018 (13).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    if normalize:
        # Nuetzi2016 (3.31) and Rucker2018 (13)
        P2 = P @ P
        return eye3 + (2 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p))
    else:
        # returns always an orthogonal matrix, but not necessary normalized,
        # see Egeland2002 (6.163)
        return (p0**2 - p @ p) * eye3 + outer3(p, 2 * p) + 2 * p0 * ax2skew(p)


@njit(cache=True)
def Exp_SO3_quat_P(P, normalize=True):
    """Derivative of Exp_SO3_quat with respect to P."""
    p0, p = P[0], P[1:]
    p_tilde = ax2skew(p)
    p_tilde_p = ax2skew_a()

    if normalize:
        P2 = P @ P
        # A_P = np.einsum(
        #     "ij,k->ijk", p0 * p_tilde + ax2skew_squared(p), -(4 / (P2 * P2)) * P
        # )
        A_P = (p0 * p_tilde + ax2skew_squared(p))[:, :, None] * (-(4 / (P2 * P2)) * P)[
            None, None, :
        ]
        s2 = 2 / P2
        A_P[:, :, 0] += s2 * p_tilde
        A_P[:, :, 1:] += (
            s2
            * p0
            * p_tilde_p
            # + np.einsum("ijl,jk->ikl", p_tilde_p, s2 * p_tilde)
            # + np.einsum("ij,jkl->ikl", s2 * p_tilde, p_tilde_p)
        )
        for i in range(3):
            m = s2 * p_tilde @ p_tilde_p[i]
            A_P[i, :, 1:] -= m
            A_P[:, :, i + 1] += m

    else:
        A_P = np.zeros((3, 3, 4), dtype=np.float64)
        A_P[:, :, 0] = 2 * p0 * eye3 + 2 * ax2skew(p)
        # A_P[:, :, 1:] = -np.multiply.outer(eye3, 2 * p) + 2 * p0 * ax2skew_a()
        A_P[:, :, 1:] -= 2 * eye3[:, :, None] * p[None, None, :]
        A_P[:, :, 1:] += 2 * p0 * ax2skew_a()
        A_P[0, :, 1:] += 2 * p[0] * eye3
        A_P[1, :, 1:] += 2 * p[1] * eye3
        A_P[2, :, 1:] += 2 * p[2] * eye3
        A_P[0, :, 1] += 2 * p
        A_P[1, :, 2] += 2 * p
        A_P[2, :, 3] += 2 * p

    return A_P


@njit(cache=True)
def T_SO3_quat(P, normalize=True):
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0], P[1:]
    if normalize:
        return (2 / (P @ P)) * np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    else:
        return 2 * (P @ P) * np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))


@njit(cache=True)
def T_SO3_inv_quat(P, normalize=True):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330), Nuetzi2016 (3.11) and (4.19) as well as Rucker2018 (21) 
    and (22).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    if normalize:
        return 0.5 * np.vstack((-p[None, :], p0 * eye3 + ax2skew(p)))
    else:
        return 1 / (2 * (P @ P) ** 2) * np.vstack((-p[None, :], p0 * eye3 + ax2skew(p)))


@njit(cache=True)
def T_SO3_quat_P(P, normalize=True):
    p0, p = P[0], P[1:]
    P2 = P @ P
    matrix = np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    if normalize:
        factor = 2 / P2
        factor_P = -4 * P / P2**2
    else:
        factor = 2 * P2
        factor_P = 4 * P

    # T_P = np.multiply.outer(matrix, factor_P)
    T_P = matrix[:, :, None] * factor_P[None, None, :]
    T_P[:, 0, 1:] -= factor * eye3
    T_P[:, 1:, 0] += factor * eye3
    T_P[:, 1:, 1:] -= factor * ax2skew_a()

    return T_P


@njit(cache=True)
def T_SO3_inv_quat_P(P, normalize=True):
    if normalize:
        T_inv_P = np.zeros((4, 3, 4), dtype=np.float64)
        T_inv_P[0, :, 1:] = -0.5 * eye3
        T_inv_P[1:, :, 0] = 0.5 * eye3
        T_inv_P[1:, :, 1:] = 0.5 * ax2skew_a()
    else:
        p0, p = P[0], P[1:]
        P2 = P @ P
        factor = 1 / (2 * P2**2)
        factor_P = -2 / (P2**3) * P
        matrix = np.vstack((-p[None, :], p0 * eye3 + ax2skew(p)))

        # T_inv_P = np.multiply.outer(matrix, factor_P)
        T_inv_P = matrix[:, :, None] * factor_P[None, None, :]
        T_inv_P[0, :, 1:] -= factor * eye3
        T_inv_P[1:, :, 0] += factor * eye3
        T_inv_P[1:, :, 1:] += factor * ax2skew_a()

    return T_inv_P


@njit(cache=True)
def quatprod(P, Q):
    """Quaternion product, see Egeland2002 (6.190).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0], P[1:]
    q0, q = Q[0], Q[1:]
    z0 = p0 * q0 - p @ q
    z = p0 * q + q0 * p + cross3(p, q)
    return np.array([z0, *z])


@njit(cache=True)
def axis_angle2quat(axis, angle):
    n = axis / norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * n])
