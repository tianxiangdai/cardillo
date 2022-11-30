import numpy as np
from math import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IK_basic, cross3, axis_angle2quat
from cardillo import System
from cardillo.discrete import Frame
from cardillo.constraints import (
    SphericalJoint,
    RevoluteJoint,
)
from cardillo.discrete import RigidBodyQuaternion
from cardillo.forces import (
    Force,
    PDRotationalJoint,
    LinearDamper,
    LinearSpring,
)
from cardillo.solver import EulerBackward, RadauIIa


def run(joint, Solver, k=None, d=None, **solver_args):
    m = 1
    r = 0.1
    l = 2
    g = 9.81

    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    C = 1 / 2 * m * r**2
    K_theta_S = np.diag(np.array([A, C, A]))

    use_spherical_joint = use_revolute_joint = use_pdrotational_joint = False

    ############################################################################
    #                   Rigid Body 1
    ############################################################################
    alpha0 = pi / 2
    alpha_dot0 = 0

    r_OB1 = np.zeros(3)
    A_IB1 = np.eye(3)
    origin = Frame(r_OP=r_OB1, A_IK=A_IB1)
    A_IK10 = A_IK_basic(alpha0).z()
    r_OS10 = -0.5 * l * A_IK10[:, 1]
    omega01 = np.array([0, 0, alpha_dot0])
    vS1 = cross3(omega01, r_OS10)
    u01 = np.concatenate([vS1, omega01])

    p01 = axis_angle2quat(np.array([0, 0, 1]), alpha0)
    q01 = np.concatenate([r_OS10, p01])
    RB1 = RigidBodyQuaternion(m, K_theta_S, q01, u01)

    if joint == "SphericalJoint":
        use_spherical_joint = True
    elif joint == "RevoluteJoint":
        use_revolute_joint = True
    elif joint == "PDRotationalJoint":
        use_pdrotational_joint = True
        assert (k and d) is not None
    else:
        raise RuntimeError(
            'Invalid Argument.\nPossible Arguments are "SphericalJoint", "RevoluteJoint", "PDRotationalJoint".'
        )

    if use_spherical_joint:
        joint1 = SphericalJoint(origin, RB1, r_OB1)
    elif use_revolute_joint:
        joint1 = RevoluteJoint(origin, RB1, r_OB1, A_IB1)
    elif use_pdrotational_joint:
        joint1 = PDRotationalJoint(
            RevoluteJoint, Spring=LinearSpring, Damper=LinearDamper
        )(origin, RB1, r_OB1, A_IB1, k=k, d=d, g_ref=-alpha0)

    ############################################################################
    #                   Rigid Body 2
    ############################################################################
    beta0 = -pi / 4
    beta_dot0 = 0

    r_OB2 = -l * A_IK10[:, 1]
    A_IB2 = A_IK10
    A_IK20 = A_IK10 @ A_IK_basic(beta0).z()
    r_B2S2 = -0.5 * l * A_IK20[:, 1]
    r_OS20 = r_OB2 + r_B2S2
    omega02 = np.array([0, 0, alpha_dot0 + beta_dot0])
    vB2 = cross3(omega01, r_OB2)
    vS2 = vB2 + cross3(omega02, r_B2S2)
    u02 = np.concatenate([vS2, omega02])

    p02 = axis_angle2quat(np.array([0, 0, 1]), alpha0 + beta0)
    q02 = np.concatenate([r_OS20, p02])
    RB2 = RigidBodyQuaternion(m, K_theta_S, q02, u02)

    if use_spherical_joint:
        joint2 = SphericalJoint(RB1, RB2, r_OB2)
    elif use_revolute_joint:
        joint2 = RevoluteJoint(RB1, RB2, r_OB2, A_IB2)
    elif use_pdrotational_joint:
        joint2 = PDRotationalJoint(
            RevoluteJoint, Spring=LinearSpring, Damper=LinearDamper
        )(RB1, RB2, r_OB2, A_IB2, k=k, d=d, g_ref=-beta0)

    ############################################################################
    #                   model
    ############################################################################
    model = System()
    model.add(origin)
    model.add(RB1)
    model.add(joint1)
    model.add(RB2)
    model.add(joint2)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB1))
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB2))

    model.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t0 = 0
    t1 = 3
    dt = 1e-2
    # dt = 5e-3
    sol = Solver(model, t1, dt, **solver_args).solve()
    t = sol.t
    q = sol.q

    ############################################################################
    #                   animation
    ############################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = 2 * l
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)
    ax.view_init(vertical_axis="y")

    def init(t, q):
        x_0, y_0, z_0 = origin.r_OP(t)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        (COM,) = ax.plot([x_0, x_S1, x_S2], [y_0, y_S1, y_S2], [z_0, z_S1, z_S2], "-ok")
        (d11_,) = ax.plot(
            [x_S1, x_S1 + d11[0]],
            [y_S1, y_S1 + d11[1]],
            [z_S1, z_S1 + d11[2]],
            "-r",
        )
        (d21_,) = ax.plot(
            [x_S1, x_S1 + d21[0]],
            [y_S1, y_S1 + d21[1]],
            [z_S1, z_S1 + d21[2]],
            "-g",
        )
        (d31_,) = ax.plot(
            [x_S1, x_S1 + d31[0]],
            [y_S1, y_S1 + d31[1]],
            [z_S1, z_S1 + d31[2]],
            "-b",
        )
        (d12_,) = ax.plot(
            [x_S2, x_S2 + d12[0]],
            [y_S2, y_S2 + d12[1]],
            [z_S2, z_S2 + d12[2]],
            "-r",
        )
        (d22_,) = ax.plot(
            [x_S2, x_S2 + d22[0]],
            [y_S2, y_S2 + d22[1]],
            [z_S2, z_S2 + d22[2]],
            "-g",
        )
        (d32_,) = ax.plot(
            [x_S2, x_S2 + d32[0]],
            [y_S2, y_S2 + d32[1]],
            [z_S2, z_S2 + d32[2]],
            "-b",
        )

        return COM, d11_, d21_, d31_, d12_, d22_, d32_

    def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
        # def update(t, q, COM, d11_, d21_, d31_):
        x_0, y_0, z_0 = origin.r_OP(t)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF], K_r_SP=np.array([0, -l / 2, 0]))
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=np.array([0, -l / 2, 0]))

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        COM.set_data([x_0, x_S1, x_S2], [y_0, y_S1, y_S2])
        COM.set_3d_properties([z_0, z_S1, z_S2])
        # COM.set_data([x_0, x_S1], [y_0, y_S1])
        # COM.set_3d_properties([z_0, z_S1])

        d11_.set_data([x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]])
        d11_.set_3d_properties([z_S1, z_S1 + d11[2]])

        d21_.set_data([x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]])
        d21_.set_3d_properties([z_S1, z_S1 + d21[2]])

        d31_.set_data([x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]])
        d31_.set_3d_properties([z_S1, z_S1 + d31[2]])

        d12_.set_data([x_S2, x_S2 + d12[0]], [y_S2, y_S2 + d12[1]])
        d12_.set_3d_properties([z_S2, z_S2 + d12[2]])

        d22_.set_data([x_S2, x_S2 + d22[0]], [y_S2, y_S2 + d22[1]])
        d22_.set_3d_properties([z_S2, z_S2 + d22[2]])

        d32_.set_data([x_S2, x_S2 + d32[0]], [y_S2, y_S2 + d32[1]])
        d32_.set_3d_properties([z_S2, z_S2 + d32[2]])

        return COM, d11_, d21_, d31_, d12_, d22_, d32_

    COM, d11_, d21_, d31_, d12_, d22_, d32_ = init(0, q[0])

    def animate(i):
        update(t[i], q[i], COM, d11_, d21_, d31_, d12_, d22_, d32_)

    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    # compute reference solution
    def _eqm(t, x):
        thetaA = A + 5 * m * (l**2) / 4
        thetaB = A + m * (l**2) / 4

        M = np.array(
            [
                [thetaA, 0.5 * m * l * l * np.cos(x[0] - x[1])],
                [0.5 * m * l * l * np.cos(x[0] - x[1]), thetaB],
            ]
        )

        h = np.array(
            [
                -0.5 * m * l * l * (x[3] ** 2) * np.sin(x[0] - x[1])
                - 1.5 * m * l * g * np.sin(x[0]),
                0.5 * m * l * l * (x[2] ** 2) * np.sin(x[0] - x[1])
                - 0.5 * m * l * g * np.sin(x[1]),
            ]
        )

        dx = np.zeros(4)
        dx[:2] = x[2:]
        dx[2:] = np.linalg.inv(M) @ h
        return dx

    def _eqm_pd_rotational(t, x):
        thetaA = A + 5 * m * (l**2) / 4
        thetaB = A + m * (l**2) / 4

        M = np.array(
            [
                [thetaA, 0.5 * m * l * l * np.cos(x[0] - x[1])],
                [0.5 * m * l * l * np.cos(x[0] - x[1]), thetaB],
            ]
        )

        h = np.array(
            [
                -0.5 * m * l * l * (x[3] ** 2) * np.sin(x[0] - x[1])
                - 1.5 * m * l * g * np.sin(x[0])
                - k * (2 * x[0] - x[1])
                - d * (2 * x[2] - x[3]),
                0.5 * m * l * l * (x[2] ** 2) * np.sin(x[0] - x[1])
                - 0.5 * m * l * g * np.sin(x[1])
                - k * (x[1] - x[0])
                - d * (x[3] - x[2]),
            ]
        )

        dx = np.zeros(4)
        dx[:2] = x[2:]
        dx[2:] = np.linalg.inv(M) @ h
        return dx

    if use_pdrotational_joint:
        eqm = _eqm_pd_rotational
    else:
        eqm = _eqm

    x0 = np.array([alpha0, alpha0 + beta0, alpha_dot0, alpha_dot0 + beta_dot0])
    ref = solve_ivp(eqm, [t0, t1], x0, method="RK45", rtol=1e-8, atol=1e-12)
    x = ref.y
    t_ref = ref.t

    alpha_ref = x[0]
    phi_ref = x[1]

    alpha = np.arctan2(sol.q[:, 0], -sol.q[:, 1])
    x_B2 = 2 * sol.q[:, 0]
    y_B2 = 2 * sol.q[:, 1]
    phi = np.arctan2(sol.q[:, 7] - x_B2, -(sol.q[:, 8] - y_B2))

    fig, ax = plt.subplots()
    ax.plot(t_ref, alpha_ref, "-.r", label="alpha ref")
    ax.plot(t, alpha, "-r", label="alpha")

    ax.plot(t_ref, phi_ref, "-.g", label="phi ref")
    ax.plot(t, phi, "-g", label="phi")

    ax.grid()
    ax.legend()

    plt.show()


if __name__ == "__main__":
    #######################
    # spherical joint tests
    #######################
    # run("SphericalJoint", EulerBackward, method="index 1")
    # run("SphericalJoint", EulerBackward, method="index 2")
    # run("SphericalJoint", EulerBackward, method="index 3")
    # run("SphericalJoint", EulerBackward, method="index 2 GGL")

    ######################
    # revolute joint tests
    ######################
    # run("RevoluteJoint", EulerBackward, method="index 1")
    # run("RevoluteJoint", EulerBackward, method="index 2")
    # run("RevoluteJoint", EulerBackward, method="index 3")
    # run("RevoluteJoint", EulerBackward, method="index 2 GGL")

    ###########################
    # PD rotational joint tests
    ###########################
    k = 1e2
    d = 1e1
    # run("PDRotationalJoint", EulerBackward, method="index 1", k=k, d=d)
    # run("PDRotationalJoint", EulerBackward, method="index 2", k=k, d=d)
    # run("PDRotationalJoint", EulerBackward, method="index 3", k=k, d=d)
    # run("PDRotationalJoint", EulerBackward, method="index 2 GGL", k=k, d=d)
    run("PDRotationalJoint", RadauIIa, k=k, d=d)
