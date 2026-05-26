"""
Single-solve static-model-based control (Li 2023) for a Dai 2025 TDCM.

Idea
----
Instead of running Newton repeatedly in a Python control loop, we
unroll the controller into Cardillo's *load-step* iteration.

    1. Each tendon's force function   la(t)   reads from a shared
       mutable state vector  lambda_t   owned by `TendonControl`.
    2. `TendonControl` is added to the System like any other
       contributor.  It declares no DOFs, but it exposes a
       `step_callback`, which Cardillo invokes after every converged
       load step.
    3. Inside that callback the controller:
           - reads the just-converged equilibrium  x = [q, λ_g, λ_c, λ_N]
             from the solver,
           - assembles  Γ = dr_OP/dλ_t  analytically via the IFT,
           - updates  lambda_t  by one forward-Euler step of
                 λ̇_t  =  -K · Γ⁺ · e .
    4. The *next* load step starts with this new lambda_t (because the
       tendon force lambdas point into our state), warm-started from
       the previous equilibrium.

Result: the entire control loop is **a single solver.solve() call**.
"""

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import splu

from cardillo.constraints import RigidConnection
from cardillo.forces import TendonForce
from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.discreteRod import DiscreteRod
from cardillo.solver import Newton, SolverOptions
from cardillo.system import System


# ----------------------------------------------------------------------
# Building blocks for Γ
# ----------------------------------------------------------------------
def assemble_W_t(system, tendons, t, q):
    """Global tendon actuation matrix W_t ∈ ℝ^(n_u × n_t).

    Column i is the generalised-force direction of a unit force in
    tendon i, scattered from local DOFs into full-system DOFs via
    tend.uDOF (= supervisor's B^T).  Sign matches  f_1 = ... + W_t λ_t.
    """
    n_t = len(tendons)
    W_t = np.zeros((system.nu, n_t))
    for i, tend in enumerate(tendons):
        q_loc = q[tend.qDOF]
        W_t[tend.uDOF, i] = -tend.W_l(t, q_loc)
    return W_t


def selection_matrix_tip(rod, n_q):
    """C ∈ ℝ^(3 × n_q) :  r_OP = C · q  (tip-node position)."""
    C = np.zeros((3, n_q))
    last = rod.nnode - 1
    pos_idx = rod.qDOF[rod.nodalDOF_r[last]]
    C[:, pos_idx] = np.eye(3)
    return C


# ----------------------------------------------------------------------
# The controller — a Cardillo contributor with no DOFs
# ----------------------------------------------------------------------
class TendonControl:
    """Static-model-based controller (Li 2023) as a System contributor.

    Has no q/u DOFs of its own.  Its purpose is twofold:
      - hold the mutable tendon-force state  self.lambda_t,
      - run one forward-Euler control update inside Cardillo's
        step_callback hook, once per converged load step.

    Attach the solver via  controller.attach_solver(solver)  *after*
    creating the Newton solver and *before* solver.solve().
    """

    # -- minimum attributes Cardillo's assemble() looks for ---------
    # We must expose nq, nu = 0 so my_qDOF/my_uDOF are empty arrays;
    # System.step_callback then passes us empty slices, which we ignore.
    nq = 0
    nu = 0
    nla_g = 0
    nla_c = 0
    nla_S = 0
    nla_N = 0

    def __init__(self, rod, tendons, u_ref, lambda_t0,
                 lambda_gain=0.1, verbose=True):
        self.rod = rod
        self.tendons = tendons
        self.u_ref_fn = u_ref if callable(u_ref) else (
            lambda t: np.asarray(u_ref, dtype=float)
        )
        self.lambda_gain = float(lambda_gain)
        self.verbose = verbose

        # mutable controller state
        self.lambda_t = np.array(lambda_t0, dtype=float)

        # Wire each tendon to read its scalar force from our state
        for i, tend in enumerate(self.tendons):
            tend.la = self._make_la_fn(i)

        # zero initial conditions for the empty DOF blocks
        self.q0  = np.zeros(0)
        self.u0  = np.zeros(0)
        self.la_g0 = np.zeros(0)
        self.la_c0 = np.zeros(0)
        self.la_N0 = np.zeros(0)
        self.la_S0 = np.zeros(0)

        # solver reference – set later from outside
        self.solver = None

        # diagnostics
        self.history = {"t": [], "lambda_t": [],
                        "u": [], "u_ref": [], "error": []}

    def _make_la_fn(self, i):
        # Each tendon's force is one component of our state vector.
        # Capture i by default-argument so the closure binds correctly.
        return lambda t, i=i: float(self.lambda_t[i])

    def attach_solver(self, solver):
        """Give the controller a handle to the Newton instance it lives
        inside, so step_callback can read the converged augmented state
        and call solver.jac()."""
        self.solver = solver

    # -----------------------------------------------------------------
    # Cardillo hooks
    # -----------------------------------------------------------------
    def assembler_callback(self):
        # Nothing to do — we own no DOFs.
        pass

    def step_callback(self, t, q_local, u_local):
        """Called by System.step_callback after each converged load step.

        q_local / u_local are empty slices (we have no DOFs).  We pull
        the full converged augmented state from self.solver instead.
        """
        if self.solver is None:
            return q_local, u_local      # not attached yet — skip

        solver = self.solver

        # 1. Locate our load step index from t -------------------------
        i = int(np.argmin(np.abs(solver.load_steps - t)))
        # Augmented state x = [q, la_g, la_c, la_N]:
        x_curr = solver.x[i].copy()
        nq = solver.system.nq
        q = x_curr[:nq]

        # 2. Tip position and error -----------------------------------
        last = self.rod.nnode - 1
        pos_idx = self.rod.qDOF[self.rod.nodalDOF_r[last]]
        r_OP = q[pos_idx].copy()
        u_r = self.u_ref_fn(t)
        e = r_OP - u_r

        # 3. Γ via IFT, using the solver's own jacobian ---------------
        Gamma = self._compute_gamma(solver, t, x_curr, q)

        # 4. Forward-Euler control update on lambda_t -----------------
        # Effective dt = spacing of the load-step grid.
        ls = solver.load_steps
        if i < len(ls) - 1:
            dt = ls[i + 1] - ls[i]
        else:
            dt = ls[-1] - ls[-2] if len(ls) > 1 else 1.0

        self.lambda_t = (
            self.lambda_t - self.lambda_gain * dt * (pinv(Gamma) @ e)
        )

        # 5. Log -------------------------------------------------------
        err_norm = float(np.linalg.norm(e))
        self.history["t"].append(float(t))
        self.history["lambda_t"].append(self.lambda_t.copy())
        self.history["u"].append(r_OP.copy())
        self.history["u_ref"].append(u_r.copy())
        self.history["error"].append(err_norm)
        if self.verbose:
            lt_str = ", ".join(f"{v:6.2f}" for v in self.lambda_t)
            print(f"  load-step {i:3d} | t={t:.3f} | "
                  f"|e|={err_norm*1e3:7.3f} mm | "
                  f"λ_t=[{lt_str}] N")

        # We did not modify q or u — return empties unchanged.
        return q_local, u_local

    # -----------------------------------------------------------------
    # Γ assembly (IFT)
    # -----------------------------------------------------------------
    def _compute_gamma(self, solver, t, x_curr, q):
        # Prime solver caches at x_curr (one residual eval), then
        # grab the augmented Jacobian.  No extra Newton iteration runs.
        solver.fun(x_curr, t)
        J = solver.jac(x_curr, t).tocsc()

        n_q = solver.system.nq
        n_u = solver.system.nu

        W_t = assemble_W_t(solver.system, self.tendons, t, q)  # (n_u, n_t)

        rhs = np.zeros((solver.nx, len(self.tendons)))
        rhs[:n_u, :] = W_t            # ∂f/∂λ_t = [W_t ; 0 ; 0 ; 0]

        lu = splu(J)
        dx_dlt = lu.solve(-rhs)        # dx/dλ_t

        C = selection_matrix_tip(self.rod, n_q)
        return C @ dx_dlt[:n_q, :]


# ----------------------------------------------------------------------
# System builder (same 4-tendon TDCM as in tendon_driven_continuum_manipulator.py)
# ----------------------------------------------------------------------
def build_system():
    rod_nelement = 1000
    rod_r_ratio  = 0.4
    rod_l_new    = 0.2
    rod_r_new    = rod_l_new * 0.05

    rod_A_IB0 = np.zeros((3, 3))
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1

    rod_m  = 0.433 * 2
    rod_r0 = 30e-3
    rod_l0 = 95e-3
    density = rod_m / (np.pi * rod_r0**2 * rod_l0)

    system = System()

    radius = lambda xi: rod_r_new * (1 - xi * (1 - rod_r_ratio))
    cross_section = CircularCrossSection(radius=radius)
    E, G = 7e5, 2e5
    EA = lambda xi: E * cross_section.area(xi)
    EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
    GA = lambda xi: G * cross_section.area(xi)
    GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
    material_model = Simo1986(
        lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
        lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
    )

    def r_OP(xi):
        return np.array([xi * rod_l_new, 0.0, 0.0])
    A_IB = lambda xi: np.eye(3)
    q0 = DiscreteRod.pose_configuration(
        rod_nelement, r_OP, A_IB, A_IB0=rod_A_IB0,
    )

    rod = DiscreteRod(
        cross_section, material_model, rod_nelement,
        Q=q0.copy(), q0=q0,
        cross_section_inertias=CrossSectionInertias(density, cross_section),
    )

    rc = RigidConnection(rod, system.origin, xi1=0)

    n_tendons = 4
    tendons = []
    for i in range(n_tendons):
        phi = np.pi * i / 2
        B_r_Mi = np.array([rod_r_new * np.cos(phi),
                           rod_r_new * np.sin(phi),
                           0.0])
        tendon = TendonForce(
            subsystem_list=[rod.get_marker(xi=0.0), rod.get_marker(xi=1.0)],
            connectivity=[(0, 1)],
            xi_list=[0.0, 1.0],
            B_r_CP_list=[B_r_Mi, B_r_Mi],
        )
        tendons.append(tendon)

    system.add(rod)
    system.add(*tendons)
    system.add(rc)
    return system, rod, tendons


# ----------------------------------------------------------------------
# Main: ONE solver.solve() drives the entire control loop
# ----------------------------------------------------------------------
if __name__ == "__main__":
    system, rod, tendons = build_system()

    # --- pick a target tip position ----------------------------------
    # (rod is along z, ~108 mm tall at lambda_t = 2N pretension)
    u_target = np.array([40e-3, 0.0, 108e-3])

    # ramp from 0 to u_target over the first ~30 % of the load steps,
    # then hold — gives the controller time to track smoothly.
    def u_ref_fn(t):
        s = min(t / 0.3, 1.0)
        return s * u_target            # start position is ≈ origin in x,y

    # --- controller --------------------------------------------------
    lambda_t0 = np.array([2.0, 2.0, 2.0, 2.0])     # pretension
    controller = TendonControl(
        rod, tendons,
        u_ref=u_ref_fn,
        lambda_t0=lambda_t0,
        lambda_gain=0.5,
        verbose=True,
    )

    system.add(controller)
    system.assemble()

    # --- ONE solver, ONE solve --------------------------------------
    n_load_steps = 100
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        verbose=False,                              # we have our own log
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),
    )
    controller.attach_solver(solver)

    print(f"Running single solver.solve() over {n_load_steps} load steps ...")
    sol = solver.solve()
    print("Done.")

    # --- quick plot of the controller history ------------------------
    try:
        import matplotlib.pyplot as plt
        h = controller.history
        t = np.asarray(h["t"])
        lam = np.asarray(h["lambda_t"])
        u   = np.asarray(h["u"])   * 1e3
        uref= np.asarray(h["u_ref"]) * 1e3
        err = np.asarray(h["error"]) * 1e3

        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for i in range(lam.shape[1]):
            ax[0].plot(t, lam[:, i], label=f"λ_{i+1}")
        ax[0].set_ylabel("Tendon force [N]"); ax[0].legend(); ax[0].grid(True)

        for j, lbl in enumerate("xyz"):
            ax[1].plot(t, u[:, j], label=f"u_{lbl}")
            ax[1].plot(t, uref[:, j], "--", alpha=0.6, label=f"u_{lbl} ref")
        ax[1].set_ylabel("Position [mm]"); ax[1].legend(ncol=2, fontsize=8); ax[1].grid(True)

        ax[2].semilogy(t, err)
        ax[2].set_ylabel("||e|| [mm]"); ax[2].set_xlabel("pseudo-time")
        ax[2].grid(True)
        fig.tight_layout()
        plt.show()
    except ImportError:
        pass
