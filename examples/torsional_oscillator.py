import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cardillo import System
from cardillo.solver import ScipyIVP, EulerBackward
from cardillo.constraints import RevoluteJoint
from cardillo.discrete import RigidBodyAxisAngle, RigidBodyQuaternion
from cardillo.forces import (
    LinearSpring,
    LinearDamper,
    PDRotationalJoint,
)
from cardillo.math import Exp_SO3, axis_angle2quat, norm


def RigidCylinder(RigidBodyParametrization):
    class _RigidCylinder(RigidBodyParametrization):
        def __init__(self, m, r, l, q0=None, u0=None):
            A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
            C = 1 / 2 * m * r**2
            K_theta_S = np.diag(np.array([A, A, C]))

            super().__init__(m, K_theta_S, q0=q0, u0=u0)

    return _RigidCylinder


def run(
    k,
    d,
    psi,
    alpha_dot0,
    g_ref=0,
    RigidBodyParametrization=RigidBodyQuaternion,
    plot=True,
):
    l = 0.1
    m = 1
    r = 0.2

    r_OP0 = np.zeros(3)
    v_P0 = np.zeros(3)
    K_Omega0 = np.array((0, 0, alpha_dot0))
    u0 = np.hstack((v_P0, K_Omega0))

    if type(RigidBodyParametrization) is type(RigidBodyAxisAngle):
        q0 = np.hstack((r_OP0, psi))
        rigid_body = RigidCylinder(RigidBodyAxisAngle)(m, r, l, q0, u0)
    elif type(RigidBodyParametrization) is type(RigidBodyQuaternion):
        n_psi = norm(psi)
        p = axis_angle2quat(psi / n_psi, n_psi)
        q0 = np.hstack((r_OP0, p))
        rigid_body = RigidCylinder(RigidBodyQuaternion)(m, r, l, q0, u0)
    else:
        raise (TypeError)

    system = System()
    joint = PDRotationalJoint(RevoluteJoint, Spring=LinearSpring, Damper=LinearDamper)(
        subsystem1=system.origin,
        subsystem2=rigid_body,
        r_OB0=np.zeros(3),
        A_IB0=A_IK0,
        k=k,
        d=d,
        g_ref=g_ref,
    )

    system.add(rigid_body, joint)
    system.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t1 = 2
    dt = 1.0e-2
    solver = ScipyIVP(system, t1, dt, atol=1e-8)
    # solver = EulerBackward(system, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    ############################################################################
    #                   plot
    ############################################################################
    if plot:
        joint.reset()
        alpha_cmp = [joint.angle(ti, qi[joint.qDOF]) for ti, qi in zip(t, q)]

        def eqm(t, x):
            dx = np.zeros(2)
            dx[0] = x[1]
            dx[1] = -2 / (m * r**2) * (d * x[1] + k * (x[0] - g_ref))
            return dx

        x0 = np.array((0, alpha_dot0))
        ref = solve_ivp(eqm, [0, t1], x0, method="RK45", rtol=1e-8, atol=1e-12)
        x = ref.y
        t_ref = ref.t
        alpha_ref = x[0]

        fig, ax = plt.subplots(1, 1)

        ax.plot(t, alpha_cmp, "-k", label="alpha")
        ax.plot(t_ref, alpha_ref, "-.r", label="alpha_ref")
        ax.legend()

        plt.show()


if __name__ == "__main__":
    profiling = False

    # initial rotational velocity e_z^K axis
    alpha_dot0 = 25
    # axis angle rotation
    # psi = np.random.rand(3)
    # psi = np.array((0, 1, 0))
    # psi = np.array((1, 0, 0))
    psi = np.array((0, 0, 1))
    # psi = np.array((0, np.pi/2, 0))
    A_IK0 = Exp_SO3(psi)

    # spring stiffness and damper parameter
    k = 1
    d = 0.025
    # k=d=0

    # Rigid body parametrization
    # RigidBodyParametrization = RigidBodyQuaternion
    RigidBodyParametrization = RigidBodyAxisAngle

    if profiling:
        import cProfile, pstats

        profiler = cProfile.Profile()

        profiler.enable()
        run(
            k=k,
            d=d,
            psi=psi,
            alpha_dot0=alpha_dot0,
            g_ref=4 * np.pi,
            RigidBodyParametrization=RigidBodyParametrization,
            plot=False,
        )
        profiler.disable()

        stats = pstats.Stats(profiler)
        # stats.print_stats(20)
        stats.sort_stats(pstats.SortKey.TIME, pstats.SortKey.CUMULATIVE).print_stats(
            0.5, "cardillo"
        )
    else:
        run(
            k=k,
            d=d,
            psi=psi,
            alpha_dot0=alpha_dot0,
            g_ref=4 * np.pi,
            RigidBodyParametrization=RigidBodyParametrization,
            plot=True,
        )
