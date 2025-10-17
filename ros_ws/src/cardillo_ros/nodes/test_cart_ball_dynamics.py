import numpy as np
from cardillo.discrete import Box, Sphere, RigidBody
from cardillo.system import System
from cardillo.forces import Force
from cardillo.constraints import FixedDistance, Prismatic
from cardillo.solver import ScipyIVP, ScipyDAE
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

m_cart = 1
m_ball = 1
gamma = m_cart / m_ball
g_accel = 9.81
l2 = 0.1
alpha0 = np.pi / 4
fps = 100

t1 = 2


# scipy ivp
M = lambda alpha: np.array(
    [
        [m_cart + m_ball, m_ball * l2 * np.cos(alpha)],
        [m_ball * l2 * np.cos(alpha), m_ball * l2**2],
    ]
)
M_inv = lambda alpha: np.array(
    [
        [m_ball * l2**2, -m_ball * l2 * np.cos(alpha)],
        [-m_ball * l2 * np.cos(alpha), m_cart + m_ball],
    ]
    / (m_cart + m_ball * np.sin(alpha) ** 2)
    / (m_ball * l2**2)
)
h = lambda alpha, dalpha: m_ball * l2 * np.sin(alpha) * np.array([dalpha**2, -g_accel])


def fun(t, y):
    x, alpha, dx, dalpha = y
    # E = 0.5 * (m_ball + m_cart) * dx**2 + 0.5 * m_ball * l2 **2 * dalpha **2 + m_ball * l2 * dx * dalpha * np.cos(alpha) - m_ball * g_accel * l2 * np.cos(alpha)
    # dE = E - m_ball * g_accel * l2
    E = 0.5 * (
        (m_cart + m_ball) - m_ball * np.cos(alpha) ** 2
    ) * l2**2 * dalpha**2 - (m_cart + m_ball) * g_accel * l2 * np.cos(alpha)
    dE = E - (m_cart + m_ball) * g_accel * l2
    if dx == 0:
        la = 0
    else:
        # la = - np.arctan(dE * dx * 1000) *2 /np.pi * 10
        la = np.arctan(dE * l2 * np.cos(alpha) * dalpha * 10) * 2 / np.pi * 10
    dy = np.empty_like(y)
    dy[:2] = y[2:]
    dy[2:] = M_inv(alpha) @ (h(alpha, dalpha) + np.array([la, 0]))
    return dy


t_eval = np.linspace(0, t1, fps * t1 + 1, endpoint=True)
s = solve_ivp(
    fun,
    method="Radau",
    t_span=[0, t1],
    y0=np.array([0, alpha0, 0, 0]),
    t_eval=t_eval,
)
x_ivp, alpha_ivp, dx_ivp, dalpha_ivp = s.y


class CartBall:
    def __init__(self):
        self.fps = fps

        self.system = System()
        self.cart = Box(RigidBody)(
            np.array([30, 10, 10]) * 1e-3, mass=m_cart, B_Theta_C=np.eye(3)
        )
        # self.force = Force(lambda t: np.array([0, 0, 0], dtype=np.float64), self.cart)
        self.ball = Sphere(RigidBody)(
            radius=10e-3,
            mass=m_ball,
            B_Theta_C=np.eye(3),
            q0=np.array([l2 * np.sin(alpha0), -l2 * np.cos(alpha0), 0, 1, 0, 0, 0]),
        )
        rc1 = Prismatic(self.cart, self.system.origin, axis=0)
        rc2 = FixedDistance(self.cart, self.ball)
        grav = Force(np.array([0, -m_ball * g_accel, 0]), self.ball)

        for el in [self.cart, self.ball, grav, rc1, rc2]:
            self.system.add(el)
        self.system.assemble()
        self.solver = ScipyIVP(self.system, t1, 1 / self.fps)

    def solve(self):
        return self.solver.solve()


cart_ball = CartBall()
solver = cart_ball.solver
# # integration time
t0 = 0
t1 = 1
dt = 1e-2
solver.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
solver.dt = dt
solver.t_eval = np.arange(t0, solver.t1 + solver.dt, solver.dt)
solver.frac = (t1 - t0) / 101
sol1 = cart_ball.solve()
# integration time
t0 = 1
dt = 1e-2
t1 = 2
solver.x0 = np.concatenate([sol1.q[-1], sol1.u[-1]])
solver.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
solver.dt = dt
solver.t_eval = np.arange(t0, solver.t1 + solver.dt, solver.dt)
solver.frac = (t1 - t0) / 101
sol2 = cart_ball.solve()

r_OS_cart = []
r_OS_ball = []
v_S_cart = []
v_S_ball = []
cart = cart_ball.cart
ball = cart_ball.ball
for si in sol1:
    ti, qi, ui = si.t, si.q, si.u
    r_OS_cart.append(cart.r_OP(ti, qi[cart.qDOF]))
    r_OS_ball.append(ball.r_OP(ti, qi[ball.qDOF]))
    v_S_cart.append(cart.v_P(ti, qi[cart.qDOF], ui[cart.uDOF]))
    v_S_ball.append(ball.v_P(ti, qi[ball.qDOF], ui[ball.uDOF]))
for si in sol2:
    ti, qi, ui = si.t, si.q, si.u
    r_OS_cart.append(cart.r_OP(ti, qi[cart.qDOF]))
    r_OS_ball.append(ball.r_OP(ti, qi[ball.qDOF]))
    v_S_cart.append(cart.v_P(ti, qi[cart.qDOF], ui[cart.uDOF]))
    v_S_ball.append(ball.v_P(ti, qi[ball.qDOF], ui[ball.uDOF]))

r_OS_cart = np.array(r_OS_cart)
r_OS_ball = np.array(r_OS_ball)
v_S_cart = np.array(v_S_cart)
v_S_ball = np.array(v_S_ball)
dr = r_OS_ball - r_OS_cart
dv = v_S_ball - v_S_cart
x, dx_ivp = r_OS_cart[:, 0], v_S_cart[:, 0]
alpha = np.arcsin(dr[:, 0] / l2)
sel = dr[:, 1] > 0
alpha[sel] = np.pi - alpha[sel]
dalpha = (
    np.linalg.norm(dv, axis=1)
    / l2
    * np.sign(np.sum(np.cross([0, 0, 1], dr) * dv, axis=1))
)


t = np.concatenate([sol1.t, sol2.t])

plt.figure()
plt.subplot(3, 3, 3)
plt.grid(True)
plt.plot(t_eval, np.rad2deg(alpha_ivp), label="scipy IVP")
plt.plot(t, np.rad2deg(alpha), label="cardillo")
plt.subplot(3, 3, 6)
plt.grid(True)
plt.plot(t_eval, np.rad2deg(dalpha_ivp), label="scipy IVP")
plt.plot(t, np.rad2deg(dalpha), label="cardillo")
plt.subplot(3, 3, 1)
plt.grid(True)
plt.plot(t_eval, x_ivp, label="scipy IVP")
plt.subplot(3, 3, 2)
plt.grid(True)
plt.plot(t_eval, x_ivp + l2 * np.sin(alpha_ivp), label="scipy IVP")
plt.subplot(3, 3, 5)
plt.grid(True)
plt.plot(t_eval, -l2 * np.cos(alpha_ivp), label="scipy IVP")

for i in range(3):
    plt.subplot(3, 3, 3 * i + 1)
    plt.plot(t, r_OS_cart[:, i], label="cardillo")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 3, 3 * i + 2)
    plt.plot(t, r_OS_ball[:, i], label="cardillo")
    plt.grid(True)
    plt.legend()

plt.show()
