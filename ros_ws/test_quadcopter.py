import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from cardillo import System
from cardillo.discrete import RigidBody, Box
from cardillo.forces import Force
from cardillo.solver import ScipyIVP
from cardillo.math import axis_angle2quat, quat2axis_angle


mass = 1.0
g_accel = 9.81
B_Theta_C = np.diag([1.0, 1.0, 3.0])
L = 0.1  # distance from center to motor
H = 0.03  # height of quadcopter body
Km = 0.01  # quadmotor moment/force ratio
input_map = np.array([[1, 1, 1, 1], [-L, L, L, -L], [-L, -L, L, L], [Km, -Km, Km, -Km]])
input_map_inv = np.linalg.inv(input_map)

r_OP0 = np.array([0.0, 0.0, 1.0])
p0 = axis_angle2quat(np.array([1.0, 1.0, 1.0]), np.deg2rad(0))
q0 = np.hstack((r_OP0, p0))

# controller gain
lam = 1
loop_rate = 20
Kd_r, Kp_r = 2 * lam, lam**2
Kd_p, Kp_p = Kd_r * loop_rate, Kp_r * loop_rate**2


class QuadMotor:
    def __init__(self, subsystem, xi=np.zeros(3), name="quad_motor"):
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

        B_r_CPs = np.array(
            [[L, -L, H / 2], [L, L, H / 2], [-L, L, H / 2], [-L, -L, H / 2]]
        )

        self.A_IB = lambda t, q: subsystem.A_IB(t, q, xi=xi)
        self.A_IB_q = lambda t, q: subsystem.A_IB_q(t, q, xi=xi)
        self.r_OP = lambda t, q, k: subsystem.r_OP(t, q, xi=xi, B_r_CP=B_r_CPs[k])
        self.J_P = lambda t, q, k: subsystem.J_P(t, q, xi=xi, B_r_CP=B_r_CPs[k])
        self.J_P_q = lambda t, q, k: subsystem.J_P_q(t, q, xi=xi, B_r_CP=B_r_CPs[k])

    def control_input(self, t, q, u):
        # trajectory
        r_OP_des, v_P_des, a_P_des = traj(t)
        # quad copter states
        r_OP, v_P = q[:3], u[:3]
        angles = R.from_rotvec(quat2axis_angle(q[3:])).as_euler("zyx", degrees=False)[
            ::-1
        ]
        gamma, beta, alpha = angles
        angles_dot = (
            np.array(
                [
                    [1, np.tan(beta) * np.sin(gamma), np.tan(beta) * np.cos(gamma)],
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma) / np.cos(beta), np.cos(gamma) / np.cos(beta)],
                ]
            )
            @ u[3:]
        )
        # position control
        B_la_sum, betad, gammad = position_control2(
            alpha, r_OP, v_P, r_OP_des, v_P_des, a_P_des
        )
        anglesd = np.array([gammad, betad, 0])
        anglesd_dot = np.zeros_like(anglesd)
        anglesd_ddot = np.zeros_like(anglesd)
        # pose control
        B_tau = pose_control(angles, angles_dot, anglesd, anglesd_dot, anglesd_ddot)
        # TODO: map to velocities is more realistic
        las = input_map_inv @ np.hstack((B_la_sum, B_tau))
        return las

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        # body fixed moment
        las = self.control_input(t, q, u)
        B_tau = input_map[1:] @ las
        h = B_tau @ self.B_J_R(t, q)
        for i in range(4):
            h += (self.A_IB(t, q) @ np.array([0, 0, las[i]])) @ self.J_P(t, q, i)
        return h

    def h_q(self, t, q, u):
        las = self.control_input(t, q, u)
        B_tau = input_map[1:] @ las
        h_q = np.einsum("i,ijk->jk", B_tau, self.B_J_R_q(t, q))
        for i in range(4):
            B_la = np.array([0, 0, las[i]])
            h += np.einsum(
                "ijk,j,il->lk", self.A_IB_q(t, q), B_la, self.J_P(t, q, i)
            ) + np.einsum("i,ijk->jk", self.A_IB(t, q) @ B_la, self.J_P_q(t, q, i))
        return h_q


def traj(t):
    r0 = np.array([0, 0, 1])
    r1 = np.array([1, 1, 2])
    t_tran = 3
    omega = np.pi / t_tran
    if t >= t_tran:
        return r1, r1 * 0, r1 * 0
    r_OP = (r1 + r0) / 2 - (r1 - r0) / 2 * np.cos(omega * t)
    v_P = (r1 - r0) / 2 * np.sin(omega * t) * omega
    a_P = (r1 - r0) / 2 * np.cos(omega * t) * omega**2
    return r_OP, v_P, a_P


def position_control2(alpha, r_OP, v_P, r_OP_des, v_P_des, a_P_des):
    a_Px, a_Py, a_Pz = a_P_des + Kd_r * (v_P_des - v_P) + Kp_r * (r_OP_des - r_OP)
    B_la_sum = mass * g_accel + mass * a_Pz
    betad = 1 / g_accel * (a_Px * np.cos(alpha) + a_Py * np.sin(alpha))
    gammad = 1 / g_accel * (a_Px * np.sin(alpha) - a_Py * np.cos(alpha))
    return B_la_sum, betad, gammad


def pose_control(angles, angles_dot, anglesd, anglesd_dot, anglesd_ddot):
    tau = (
        1
        / np.diagonal(B_Theta_C)
        * (anglesd_ddot + Kd_p * (anglesd_dot - angles_dot) + Kp_p * (anglesd - angles))
    )
    return tau


system = System()
quadcopter = Box(RigidBody)(
    dimensions=[2 * L, 2 * L, H],
    mass=mass,
    B_Theta_C=B_Theta_C,
    q0=q0,
    name="quadcopter",
)
gravity = Force(np.array([0.0, 0.0, -mass * g_accel]), quadcopter, name="gravity")

quad_motor = QuadMotor(quadcopter)

system.add(quadcopter)
system.add(gravity)
system.add(quad_motor)
system.assemble()

solver = ScipyIVP(system, t1=10, dt=1e-2)
sol = solver.solve()

r_OP = sol.q[:, :3]
v_P = sol.u[:, :3]
A_IB = []
euler_angles = []
for sol_i in sol:
    ti, qi, ui = sol_i.t, sol_i.q, sol_i.u
    A = quadcopter.A_IB(ti, qi[quadcopter.qDOF])
    A_IB.append(A)
    euler_angles.append(R.from_matrix(A).as_euler("zyx", degrees=True)[::-1])
A_IB = np.array(A_IB)
euler_angles = np.array(euler_angles)

# trajectory
r_OP_des, v_P_des, a_P_des = np.swapaxes(np.stack([traj(ti) for ti in sol.t]), 0, 1)

beta_des, gamma_des = np.rad2deg(
    np.stack(
        [
            position_control2(al, r, v, rd, vd, ad)[1:]
            for al, r, v, rd, vd, ad in zip(
                euler_angles[:, 2], r_OP, v_P, r_OP_des, v_P_des, a_P_des
            )
        ]
    )
).T


fig, ax = plt.subplots(3, 2, sharex=True)
ax[0, 0].plot(sol.t, r_OP_des[:, 0], "-r", label="x_des")
ax[0, 0].plot(sol.t, r_OP[:, 0], label="x")
ax[1, 0].plot(sol.t, r_OP_des[:, 1], "-r", label="y_des")
ax[1, 0].plot(sol.t, r_OP[:, 1], label="y")
ax[2, 0].plot(sol.t, r_OP_des[:, 2], "-r", label="z_des")
ax[2, 0].plot(sol.t, r_OP[:, 2], label="z")
ax[0, 1].plot(sol.t, euler_angles[:, 0], label="pitch")
ax[0, 1].plot(sol.t, gamma_des, "-r", label="pitch_des")
ax[1, 1].plot(sol.t, euler_angles[:, 1], label="roll")
ax[1, 1].plot(sol.t, beta_des, "-r", label="roll_des")
ax[2, 1].plot(sol.t, euler_angles[:, 2], label="yaw")
# ax[2, 1].plot(sol.t, alpha_des, "-r", label='yaw_des')
plt.show()
