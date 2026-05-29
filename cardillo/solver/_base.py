import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu

from cardillo.definitions import IS_CLOSE_ATOL
from .solver_options import SolverOptions


def consistent_initial_conditions(
    system,
    slice_active_contacts=True,
    options=SolverOptions(),
):
    """Checks consistency of initial conditions with constraints on position and velocity level and finds initial accelerations and constraint/contact forces.

    Parameters
    ----------
    system : cardillo.System
        System for which the consistent initial conditions are computed.
    slice_active_contacts : bool
        Slice friction forces to contemplate only those corresponding to active normal contact.
    options : cardillo.solver.SolverOptions
        Solver options for the computations of the constraint/contact forces.
    """
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    q0, u0 = system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

    if (
        not options.compute_consistent_initial_conditions or system.nu == 0
    ):  # second case can happen during debugging, when only frames are added to the system
        return (
            t0,
            q0,
            u0,
            q_dot0,
            np.zeros(system.nu),
            np.zeros(system.nla_g),
            np.zeros(system.nla_c),
        )

    # evaluate constant quantities
    M = system.M(t0, q0)
    h = system.h(t0, q0, u0)

    W_g = system.W_g(t0, q0)
    g_dot_u = system.g_dot_u(t0, q0)
    zeta_g = system.zeta_g(t0, q0, u0)

    W_c = system.W_c(t0, q0)
    la_c0 = system.la_c(t0, q0, u0)


    split_x = np.cumsum(
        [
            system.nu,
            system.nla_g,
        ]
    )[:-1]

    # fmt: off
    A = bmat(
        [
            [      M, -W_g],
            [g_dot_u, None],
        ],
        format="csc",
    )
    # fmt: on

    lu = splu(A)

    b0 = np.concatenate(
        [
            h + W_c @ la_c0,
            -zeta_g,
        ]
    )


    x0 = np.zeros(system.nu + system.nla_g)

    # compute accelerations and constraints without contacts
    x0 = lu.solve(b0)


    u_dot0, la_g0 = np.array_split(x0, split_x)

    # check if initial conditions satisfy constraints on position, velocity
    # and acceleration level
    g0 = system.g(t0, q0)
    g_dot0 = system.g_dot(t0, q0, u0)
    g_ddot0 = system.g_ddot(t0, q0, u0, u_dot0)
    g_S0 = system.g_S(t0, q0)

    assert np.allclose(
        g0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g0!"
    assert np.allclose(
        g_dot0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_dot0!"
    assert np.allclose(
        g_ddot0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_ddot0!"
    assert np.allclose(
        g_S0, np.zeros(system.nla_S), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_S0!"

    return t0, q0, u0, q_dot0, u_dot0, la_g0, la_c0
