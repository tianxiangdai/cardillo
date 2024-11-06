import numpy as np
from os import path
from scipy.integrate import odeint
from matplotlib import pyplot as plt

from cardillo import System
from cardillo.math import axis_angle2quat, cross3
from cardillo.discrete import RigidBody, Meshed
from cardillo.solver import ScipyIVP, ScipyDAE
from cardillo.constraints import Revolute
from cardillo.forces import Force
from cardillo.constraints import GearTransmission
from cardillo.visualization import Renderer

# parameter
m1 = (445 + 124) * 1e-3
theta_A = (29644.7665 * 445 / 409.3892 + 190.6349 * 124 / 78.5229) * 1e-7

m2 = (64 + 45) * 1e-3
theta_B = (27.0383 * 64 / 12.0674 + 181) * 1e-7

m3 = 537 * 1e-3
theta_C = (17462.29 * 537 / 458.7439) * 1e-7

l1 = 5.04e-3
l2 = 70e-3
l3 = 150e-3

radius_B = 13e-3
radius_C = 9e-3
eta = radius_B / radius_C

z_offset = -10e-3

g = 9.81

kp, kd = np.array([2.5261, 0.0032]), np.array([0.3712, 0.0070])

tend = 10
dt = 1e-2
alpha0 = np.deg2rad(170)


class Controller:
    def __init__(self, subsystem1: Revolute, subsystem2: Revolute, kp, kd, tau):
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.kp = kp
        self.kd = kd
        if callable(tau):
            self.tau = tau
        else:
            self.tau = lambda t: tau
        self.nla_tau = 1
        self.ntau = 4

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        self._nq1 = self.subsystem1._nq
        self._nq2 = self.subsystem2._nq
        self._nq = self._nq1 + self._nq2
        self.qDOF = np.concatenate((qDOF1, qDOF2))

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        self._nu1 = self.subsystem1._nu
        self._nu2 = self.subsystem2._nu
        self._nu = self._nu1 + self._nu2
        self.uDOF = np.concatenate((uDOF1, uDOF2))

    def Wla_tau_q(self, t, q, u):
        return np.einsum(
            "ijk,j->ik", self.W_tau_q(t, q), self.la_tau(t, q, u)
        ) + self.W_tau(t, q) @ self.la_tau_q(t, q, u)

    def Wla_tau_u(self, t, q, u):
        return self.W_tau(t, q) @ self.la_tau_u(t, q, u)

    def W_tau(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        W_tau = np.zeros((self._nu, self.nla_tau), float)
        W_tau[nu1:, :] = -self.subsystem2.W_l(t, q[nq1:])
        return W_tau

    def W_tau_q(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        W_tau_q = np.zeros((self._nu, self.nla_tau, self._nq), float)
        W_tau_q[nu1:, :, nq1:] = -self.subsystem2.W_l_q(t, q[nq1:])
        return W_tau_q

    def la_tau(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        return -np.array(
            [
                self.kp[0] * (self.subsystem1.l(t, q[:nq1]) - self.tau(t)[0])
                + self.kd[0]
                * (self.subsystem1.l_dot(t, q[:nq1], u[:nu1]) - self.tau(t)[1])
                + self.kp[1] * (self.subsystem2.l(t, q[nq1:]) - self.tau(t)[2])
                + self.kd[1]
                * (self.subsystem2.l_dot(t, q[nq1:], u[nu1:]) - self.tau(t)[3])
            ]
        )

    def la_tau_q(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        la_tau_q = np.zeros((self.nla_tau, self._nq), float)
        la_tau_q[:, :nq1] = -self.kp[0] * self.subsystem1.l_q(t, q[:nq1]) - self.kd[
            0
        ] * self.subsystem1.l_dot_q(t, q[:nq1], u[:nu1])
        la_tau_q[:, nq1:] = -self.kp[1] * self.subsystem2.l_q(t, q[nq1:]) - self.kd[
            1
        ] * self.subsystem2.l_dot_q(t, q[nq1:], u[nu1:])
        return la_tau_q

    def la_tau_u(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        la_tau_u = np.zeros((self.nla_tau, self._nu), float)
        la_tau_u[:, :nu1] = -self.kd[0] * self.subsystem1.l_dot_u(t, q[:nq1], u[:nu1])
        la_tau_u[:, nu1:] = -self.kd[1] * self.subsystem2.l_dot_u(t, q[nq1:], u[nu1:])
        return la_tau_u


M = np.array(
    [
        [
            m1 * l1**2 + theta_A + m2 * l2**2 + theta_B + m3 * l3**2 + theta_C,
            theta_B + eta * theta_C,
        ],
        [theta_B + eta * theta_C, theta_B + eta**2 * theta_C],
    ]
)

c0 = (-m1 * l1 + m2 * l2 - m3 * l3) * g

# build system
system = System()

# pendulum
r_OS = np.array([l1 * np.sin(alpha0), -l1 * np.cos(alpha0), z_offset])
phi = alpha0
p = axis_angle2quat(np.array([0, 0, 1]), phi)
omega = np.array([0, 0, 0])
v_S = cross3(omega, r_OS)
q0_pendulum = np.concatenate([r_OS, p])
u0 = np.concatenate([v_S, omega])
pendulum = Meshed(RigidBody)(
    path.join(path.dirname(__file__), "stl", "Pendulum.STL"),
    scale=1e-3,
    B_r_CP=np.array([0, l1, 0]),
    mass=m1,
    B_Theta_C=np.diag([1, 1, theta_A]),
    q0=q0_pendulum,
    name="pendulum",
)
system.add(pendulum)

# rotor
r_OS = np.array([-l2 * np.sin(alpha0), l2 * np.cos(alpha0), z_offset])
phi = alpha0
p = axis_angle2quat(np.array([0, 0, 1]), phi)
omega = np.array([0, 0, 0])
v_S = cross3(omega, r_OS)
q0_rotor = np.concatenate([r_OS, p])
u0 = np.concatenate([v_S, omega])
rotor = Meshed(RigidBody)(
    path.join(path.dirname(__file__), "stl", "Rotor.STL"),
    scale=1e-3,
    B_r_CP=np.array([0, -l2, 0]),
    mass=m2,
    B_Theta_C=np.diag([1, 1, theta_B]),
    q0=q0_rotor,
    name="rotor",
)
system.add(rotor)

# flywheel
r_OS = np.array([l3 * np.sin(alpha0), -l3 * np.cos(alpha0), z_offset])
phi = alpha0
p = axis_angle2quat(np.array([0, 0, 1]), phi)
omega = np.array([0, 0, 0])
v_S = cross3(omega, r_OS)
q0_flywheel = np.concatenate([r_OS, p])
u0 = np.concatenate([v_S, omega])
flywheel = Meshed(RigidBody)(
    path.join(path.dirname(__file__), "stl", "Flywheel.STL"),
    scale=1e-3,
    B_r_CP=np.array([0, l3, 0]),
    mass=m3,
    B_Theta_C=np.diag([1, 1, theta_C]),
    q0=q0_flywheel,
    name="flywheel",
)

system.add(flywheel)

rj1 = Revolute(system.origin, pendulum, axis=2, angle0=alpha0)
system.add(rj1)

rj2 = Revolute(pendulum, rotor, axis=2, r_OJ0=q0_rotor[:3], angle0=0)
system.add(rj2)

rj3 = Revolute(pendulum, flywheel, axis=2, r_OJ0=q0_flywheel[:3], angle0=0)
system.add(rj3)

gear = GearTransmission(rj2, rj3, radius_B, radius_C)
system.add(gear)

for body in [pendulum, rotor, flywheel]:
    system.add(Force(np.array([0, -body.mass * g, 0]), body))

system.add(Controller(rj1, rj2, kp, kd, tau=[np.pi, 0, 0, 0]))
# system.add(PDcontroller(rj1, 2.5261, 0.3712, lambda t: [np.pi, 0]))
system.assemble()

solver = ScipyDAE(system, tend, dt)
sol = solver.solve()

ren = Renderer(system, [flywheel, rotor, pendulum])
ren.render_solution(sol, repeat=True)

rj1.previous_quadrant = 1
rj1.n_full_rotations = 0
rj2.previous_quadrant = 1
rj2.n_full_rotations = 0
angle = np.zeros((len(sol.t), 2), sol.q.dtype)
for i, (ti, qi) in enumerate(zip(sol.t, sol.q)):
    angle[i] = rj1.angle(ti, qi[rj1.qDOF]), rj2.angle(ti, qi[rj2.qDOF])
g = np.array([gear.g(ti, qi[gear.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])


# analytical solution
def func(x, t):
    tau = (
        kp[0] * (np.pi - x[0])
        + kp[1] * (0 - x[1])
        + kd[0] * (0 - x[2])
        + kd[1] * (0 - x[3])
    )
    return np.array([*x[2:], *np.linalg.solve(M, [c0 * np.sin(x[0]), -tau])])


t_ref = np.linspace(0, tend, int(tend / dt))
y0 = np.array([alpha0, 0, 0, 0])
sol_ref = odeint(func=func, y0=y0, t=t_ref)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_ref, sol_ref[:, 0] * 180 / np.pi, "r", label="analytical")
plt.plot(sol.t, angle[:, 0] * 180 / np.pi, "--", label="cardillo")
plt.ylabel("alpha")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t_ref, sol_ref[:, 1] * 180 / np.pi, "r")
plt.plot(sol.t, angle[:, 1] * 180 / np.pi, "--")
plt.ylabel("beta")
plt.xlabel("time")
plt.show()
