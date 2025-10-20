import numpy as np
from cardillo.discrete import Box, Sphere, RigidBody
from cardillo.system import System
from cardillo.forces import Force
from cardillo.constraints import FixedDistance, Prismatic
from cardillo.solver import ScipyIVP, ScipyDAE
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

test_cardillo = True

m_cart = 1
m_pole = 1
gamma = m_cart / m_pole
g_accel = 9.81
l2 = 0.1
alpha0 = np.pi / 4
fps = 100
t1 = 2


# scipy ivp
M = lambda alpha: np.array(
    [
        [m_cart + m_pole, m_pole * l2 * np.cos(alpha)],
        [m_pole * l2 * np.cos(alpha), m_pole * l2**2],
    ]
)
M_inv = lambda alpha: np.array(
    [
        [m_pole * l2**2, -m_pole * l2 * np.cos(alpha)],
        [-m_pole * l2 * np.cos(alpha), m_cart + m_pole],
    ]
    / (m_cart + m_pole * np.sin(alpha) ** 2)
    / (m_pole * l2**2)
)
h = lambda alpha, dalpha: m_pole * l2 * np.sin(alpha) * np.array([dalpha**2, -g_accel])


def fun(t, y):
    x, alpha, dx, dalpha = y
    # E = 0.5 * (m_pole + m_cart) * dx**2 + 0.5 * m_pole * l2 **2 * dalpha **2 + m_pole * l2 * dx * dalpha * np.cos(alpha) - m_pole * g_accel * l2 * np.cos(alpha)
    # dE = E - m_pole * g_accel * l2
    E = 0.5 * (
        (m_cart + m_pole) - m_pole * np.cos(alpha) ** 2
    ) * l2**2 * dalpha**2 - (m_cart + m_pole) * g_accel * l2 * np.cos(alpha)
    dE = E - (m_cart + m_pole) * g_accel * l2
    if dx == 0 or test_cardillo:
        la = 0
    else:
        # la = - np.arctan(dE * dx * 1000) *2 /np.pi * 10
        la = np.arctan(dE * l2 * np.cos(alpha) * dalpha * 10) * 2 / np.pi * 10
    dy = np.empty_like(y)
    dy[:2] = y[2:]
    dy[2:] = M_inv(alpha) @ (h(alpha, dalpha) + np.array([la, 0]))
    return dy


class CartPole:
    def __init__(self):
        self.fps = fps

        self.system = System()
        self.cart = Box(RigidBody)(
            np.array([30, 10, 10]) * 1e-3, mass=m_cart, B_Theta_C=np.eye(3)
        )
        # self.force = Force(lambda t: np.array([0, 0, 0], dtype=np.float64), self.cart)
        self.pole = Sphere(RigidBody)(
            radius=10e-3,
            mass=m_pole,
            B_Theta_C=np.eye(3),
            q0=np.array([l2 * np.sin(alpha0), -l2 * np.cos(alpha0), 0, 1, 0, 0, 0]),
        )
        rc1 = Prismatic(self.cart, self.system.origin, axis=0)
        rc2 = FixedDistance(self.cart, self.pole)
        grav = Force(np.array([0, -m_pole * g_accel, 0]), self.pole)

        for el in [self.cart, self.pole, grav, rc1, rc2]:
            self.system.add(el)
        self.system.assemble()
        self.solver = ScipyIVP(self.system, t1, 1 / self.fps)

    def set_solver(self, t0, t1, dt, x0=None):
        solver = self.solver
        solver.x0 = solver.x0 if x0 is None else x0
        solver.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        solver.dt = dt
        solver.t_eval = np.arange(t0, solver.t1 + solver.dt, solver.dt)
        solver.frac = (t1 - t0) / 101

    def solve(self):
        return self.solver.solve()


if test_cardillo:
    cart_pole = CartPole()
    solver = cart_pole.solver
    # # integration time
    t0 = 0
    t1 = 1
    dt = 1e-2
    cart_pole.set_solver(t0, t1, dt)
    sol1 = cart_pole.solve()
    # integration time
    t0 = 1
    dt = 1e-2
    t1 = 2
    x0 = np.concatenate([sol1.q[-1], sol1.u[-1]])
    cart_pole.set_solver(t0, t1, dt, x0)
    sol2 = cart_pole.solve()

    r_OS_cart = []
    r_OS_pole = []
    v_S_cart = []
    v_S_pole = []
    cart = cart_pole.cart
    pole = cart_pole.pole
    for si in sol1:
        ti, qi, ui = si.t, si.q, si.u
        r_OS_cart.append(cart.r_OP(ti, qi[cart.qDOF]))
        r_OS_pole.append(pole.r_OP(ti, qi[pole.qDOF]))
        v_S_cart.append(cart.v_P(ti, qi[cart.qDOF], ui[cart.uDOF]))
        v_S_pole.append(pole.v_P(ti, qi[pole.qDOF], ui[pole.uDOF]))
    for si in sol2:
        ti, qi, ui = si.t, si.q, si.u
        r_OS_cart.append(cart.r_OP(ti, qi[cart.qDOF]))
        r_OS_pole.append(pole.r_OP(ti, qi[pole.qDOF]))
        v_S_cart.append(cart.v_P(ti, qi[cart.qDOF], ui[cart.uDOF]))
        v_S_pole.append(pole.v_P(ti, qi[pole.qDOF], ui[pole.uDOF]))

    r_OS_cart = np.array(r_OS_cart)
    r_OS_pole = np.array(r_OS_pole)
    v_S_cart = np.array(v_S_cart)
    v_S_pole = np.array(v_S_pole)
    dr = r_OS_pole - r_OS_cart
    dv = v_S_pole - v_S_cart
    x, dx_ivp = r_OS_cart[:, 0], v_S_cart[:, 0]
    alpha = np.arcsin(dr[:, 0] / l2)
    sel = dr[:, 1] > 0
    alpha[sel] = np.pi - alpha[sel]
    dalpha = (
        np.linalg.norm(dv, axis=1)
        / l2
        * np.sign(np.sum(np.cross([0, 0, 1], dr) * dv, axis=1))
    )


t_eval = np.linspace(0, t1, fps * t1 + 1, endpoint=True)
s = solve_ivp(
    fun,
    method="Radau",
    t_span=[0, t1],
    y0=np.array([0, alpha0, 0, 0]),
    t_eval=t_eval,
)
x_ivp, alpha_ivp, dx_ivp, dalpha_ivp = s.y


plt.figure()
if test_cardillo:
    t = np.concatenate([sol1.t, sol2.t])
    plt.subplot(3, 3, 3)
    plt.plot(t, np.rad2deg(alpha), label="cardillo")
    plt.subplot(3, 3, 6)
    plt.plot(t, np.rad2deg(dalpha), label="cardillo")
    for i in range(3):
        plt.subplot(3, 3, 3 * i + 1)
        plt.plot(t, r_OS_cart[:, i], label="cardillo")
        plt.grid(True)
        plt.subplot(3, 3, 3 * i + 2)
        plt.plot(t, r_OS_pole[:, i], label="cardillo")
        plt.grid(True)
plt.subplot(3, 3, 3)
plt.grid(True)
plt.plot(t_eval, np.rad2deg(alpha_ivp), "--", label="scipy IVP")
plt.legend()
plt.subplot(3, 3, 6)
plt.grid(True)
plt.plot(t_eval, np.rad2deg(dalpha_ivp), "--", label="scipy IVP")
plt.legend()
plt.subplot(3, 3, 1)
plt.grid(True)
plt.plot(t_eval, x_ivp, "--", label="scipy IVP")
plt.legend()
plt.subplot(3, 3, 2)
plt.grid(True)
plt.plot(t_eval, x_ivp + l2 * np.sin(alpha_ivp), "--", label="scipy IVP")
plt.legend()
plt.subplot(3, 3, 5)
plt.grid(True)
plt.plot(t_eval, -l2 * np.cos(alpha_ivp), "--", label="scipy IVP")
plt.legend()


plt.show()
