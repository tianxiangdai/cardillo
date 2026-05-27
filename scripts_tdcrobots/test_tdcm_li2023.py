from cardillo.math import A_IB_basic
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, TendonForce
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.discreteRod import DiscreteRod

from cardillo.solver import ScipyDAE, Newton, SolverOptions
from cardillo.system import System

from cardillo.interactions import nPointInteraction

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import splu


class TendonControl :
    nq = 0
    nu = 0

    def __init__(self, rod, tendons, u_ref, lambda_t0,
                 lambda_gain=0.1, settle_steps=20, verbose=True):
        self.rod = rod
        self.tendons = tendons
        self.u_ref_fn = u_ref if callable(u_ref) else (
            lambda t: np.asarray(u_ref, dtype=float)
        )
        self.lambda_gain = float(lambda_gain)
        self.settle_steps = int(settle_steps)
        self.verbose = verbose
        self.gamma_0_inv = None

        # mutable controller state
        self.lambda_t = np.array(lambda_t0, dtype=float)

        # Wire each tendon to read its scalar force from our state
        for i, tend in enumerate(self.tendons):
            tend.la = self._make_la_fn(i)

        # zero initial conditions for the empty DOF blocks
        self.q0  = np.zeros(0)
        self.u0  = np.zeros(0)

        # solver reference – set later from outside
        self.solver = None

        # diagnostics
        self.history = {"t": [], "lambda_t": [],
                        "u": [], "u_ref": [], "error": []}
    
    def _make_la_fn(self, i):
        return lambda t, i=i: float(self.lambda_t[i])
    
    def attach_solver(self, solver):
        self.solver = solver
    
    def compute_gamma(self, solver, t, x_curr, q):
        solver.fun(x_curr, t)
        J = solver.jac(x_curr, t).tocsc()
        nq = solver.system.nq
        nu = solver.system.nu

        W_t = assemble_W_t(solver.system, self.tendons, t, q)

        rhs = np.zeros((solver.nx, len(self.tendons)))
        rhs[ :nu, : ] = W_t

        lu = splu(J)
        dx_dlambda_t = lu.solve(-rhs)

        C = selection_matrix(self.rod, nq)
        return C @ dx_dlambda_t[:nq, :]
    
    def assembler_callback(self):
        pass

    def step_callback(self, t, q_local, u_local):
        if self.solver is None:
            return q_local, u_local
        solver = self.solver

        # locate load step index from t
        i = int(np.argmin(np.abs(solver.load_steps - t)))
        # Augmented state x = [q, la_g, la_c, la_N]:
        x_curr = solver.x[i].copy()
        nq = solver.system.nq
        q = x_curr[:nq]

        # 2. Tip position and error
        last = self.rod.nnode - 1
        pos_idx = self.rod.qDOF[self.rod.nodalDOF_r[last]]
        r_OP = q[pos_idx].copy()

        # time step size
        ls = solver.load_steps
        # if i < len(ls) - 1:
        #     dt = ls[i + 1] - ls[i]
        # else:
        #     dt = ls[-1] - ls[-2] if len(ls) > 1 else 0.01

        # Settling phase for initial configuration
        if i < self.settle_steps:
            e_E = r_OP - TABLE_II["E"]
            gamma_now = self.compute_gamma(solver, t, x_curr, q)
            # settle_gain = 0.2
            settle_gain = 50.0
            # gamma_now_inv = pinv(gamma_now)
            gamma_now_inv = gamma_now.T # pseudo-inverse
            # self.lambda_t = np.maximum(self.lambda_t - settle_gain * dt * (gamma_now_inv @ e_E), 0.1)
            self.lambda_t = np.maximum(self.lambda_t - settle_gain * (gamma_now_inv @ e_E), 0.1)
            print(f" |e_E|={np.linalg.norm(e_E)*1e3:.3f} mm | ")
            return q_local, u_local

        # 3. Compute Γ = ∂r_OP/∂λ_t
        # if self.gamma_0_inv is None:
        if i == self.settle_steps or self.gamma_0_inv is None:
            gamma_0 = self.compute_gamma(solver, t, x_curr, q)
            print(f"cond(Γ₀) = {np.linalg.cond(gamma_0):.2e}")
            # self.gamma_0_inv = pinv(gamma_0)
            self.gamma_0_inv = gamma_0.T # pseudo-inverse

        # local time
        t_local = t - ls[self.settle_steps]
        u_r = self.u_ref_fn(t_local)
        e = r_OP - u_r

        lambda_min = 0.1 # tendons push only
        # self.lambda_t = np.maximum(self.lambda_t - self.lambda_gain * dt * (self.gamma_0_inv @ e), lambda_min)
        self.lambda_t = np.maximum(self.lambda_t - self.lambda_gain * (self.gamma_0_inv @ e), lambda_min)

        delta_bound = 0.5
        delta = -self.lambda_gain * (self.gamma_0_inv @ e)
        delta = np.clip(delta, -delta_bound, delta_bound) # limit per-step change
        self.lambda_t = np.maximum(self.lambda_t + delta, lambda_min)

        # 5. Logging
        err_norm = float(np.linalg.norm(e))
        self.history["t"].append(float(t_local))
        self.history["lambda_t"].append(self.lambda_t.copy())
        self.history["u"].append(r_OP.copy())
        self.history["u_ref"].append(u_r.copy())
        self.history["error"].append(err_norm)
        if self.verbose:
            lt_str = ", ".join(f"{v:6.2f}" for v in self.lambda_t)
            print(f"  load-step {i:3d} | t={t:.3f} | "
                  f"|e|={err_norm*1e3:7.3f} mm | "
                  f"λ_t=[{lt_str}] N")
            
        return q_local, u_local

def selection_matrix(rod, n_q):
    C = np.zeros((3, n_q))
    last = rod.nnode - 1
    pos_idx = rod.qDOF[rod.nodalDOF_r[last]]
    C[:, pos_idx] = np.eye(3)
    return C

def assemble_W_t(system, tendons, t, q):
    W_t = np.zeros((system.nu, len(tendons)))
    for i, tendon in enumerate(tendons):
        q_loc = q[tendon.qDOF]
        W_t[tendon.uDOF, i] = - tendon.W_l(t, q_loc) #check sign convention
    return W_t

# ---- parameters ----
rod_nelement = 500 # 1000
rod_l0 = 0.192 # [m] length of rod
rod_r0_base = 1.4e-2 # [m] radius at bottom of rod
rod_r0_tip = 8.5e-3 # [m] radius at tip of rod
density = 1.41e3 # density of material
rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1

scaling_factor = 1.0
rod_l0_new = rod_l0 * scaling_factor
rod_r0_base_new = rod_r0_base * scaling_factor
rod_r0_tip_new = rod_r0_tip * scaling_factor

# ---- rod ----
radius = lambda xi: rod_r0_base_new * (1 - xi) + rod_r0_tip_new * xi
cross_section = CircularCrossSection(radius)
E, G = 2.563e5, 8.543e4 
EA = lambda xi: E * cross_section.area(xi)
EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
GA = lambda xi: G * cross_section.area(xi)
GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
material_model = Simo1986(
    lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
    lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
)

# ---- system ----
system = System()

# inital configuration
Rod = DiscreteRod

def r_OP(xi):
    return np.array([xi*rod_l0_new, 0, 0], dtype=np.float64)

A_IB = lambda xi: np.eye(3, dtype=np.float64)
q0 = Rod.pose_configuration(
    rod_nelement,
    r_OP,
    A_IB,
    A_IB0=rod_A_IB0,
)
Q = q0.copy()

rod = Rod(
    cross_section,
    material_model,
    rod_nelement,
    Q=Q,
    q0=q0,
    cross_section_inertias=CrossSectionInertias(density, cross_section),
)

# ---- rigid connections ----
rc = RigidConnection(rod, system.origin, xi1=0)

# ---- tendons ----
n_tendons = 4
tendons = []
B_r_CP_lists = [
    [
        rod_A_IB0.T
        @ np.array(
            [
                radius(xi) * np.cos(phi),
                radius(xi) * np.sin(phi),
                0,
            ]
        )
        for xi in np.linspace(0, 1, rod_nelement + 1)
    ]
    for phi in np.linspace(0, 2 * np.pi, n_tendons, endpoint=False)
]
for B_r_CP_list in B_r_CP_lists:
    n = len(B_r_CP_list)
    tendon = TendonForce(
        subsystem_list=[rod.get_marker(i / (n - 1)) for i in range(n)],
        connectivity=[(i, i + 1) for i in range(n - 1)],
        xi_list=[i / (n - 1) for i in range(n)],
        B_r_CP_list=B_r_CP_list,
    )
    tendons.append(tendon)

# ---- controller ----

# u_target = np.array([0.15438, 0.04335, 0.03399])
# Table II from Paper
TABLE_II = {
    "A": np.array([15.438e-2,  4.335e-2,  3.399e-2]),
    "B": np.array([15.272e-2, -5.114e-2, -0.463e-2]),
    "C": np.array([10.888e-2,  9.106e-2, -5.492e-2]),
    "D": np.array([14.615e-2, -4.486e-2, -6.375e-2]),
    "E": np.array([13.951e-2,  0.000e-2, -9.842e-2]),
}
def paper_to_cardillo(u):
    X, Y, Z = u
    return np.array([Y, Z, X])
TABLE_II = {k: paper_to_cardillo(u) for k, u in TABLE_II.items()}
SEQUENCE = [ "A", "B", "C", "D", "E"]
hold_t = 1.0 / (len(SEQUENCE))
# total_t = hold_t * len(SEQUENCE) # 50 s

def u_ref_fn(t):
        k = min(int(t/hold_t), len(SEQUENCE) - 1)
        return TABLE_II[SEQUENCE[k]]

lambda_t0 = np.array([1.0, 1.0, 1.0, 1.0]) * 0.5 # causes rank deficiency in gamma_0
# lambda_t0 = np.array([4.0, 1.0, 1.0, 4.0])
# lambda_gain = 0.2
lambda_gain = 50.0
settle_steps = 30
controller = TendonControl(rod, tendons, u_ref = u_ref_fn, lambda_t0 = lambda_t0, lambda_gain=lambda_gain,settle_steps=settle_steps,verbose=True)

# ---- add to system ----
system.add(rod, rc, *tendons,controller)
system.assemble()

#################
# visualization #
#################
# ---- visual objects ----
from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon

VisualDiscreteRod(rod, subdivision=4, opacity=0.3)
for tendon in tendons:
    VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))  # (130, 130, 130),
# VisualCoordSystem(system.origin, 0.05)
# ---- plotter ----
window_size = (960, 540)
plotter = Plotter(system, window_size)
x0, x1 = -0.2, 0.2
y0, y1 = -0.2, 0.2
res_x = res_y = 10
plotter.add_ground(x0, x1, y0, y1, res_x, res_y)
# ---- camera pose ----
r_OC = np.array([0, -0.35, 0.1], float)
# r_OC = np.array([0, -0.35, 0.15], float)
r_OF = np.array([0, 0, 0.06], float)  # camera focal point
e_x_cam = np.array([1, 0, 0], float)
e_z_cam = r_OF - r_OC
e_z_cam /= np.linalg.norm(e_z_cam)
e_y_cam = np.cross(e_z_cam, e_x_cam)
zoom = 1
# zoom = 1.5
fx = fy = 2635.5177
px, py = 3840, 2160  # camera 4k resolution
cam_view_angle = np.rad2deg(np.arctan(min(px, py) / 2 / fx) * 2)
cam = plotter.camera
cam.view_angle = cam_view_angle
cam.parallel_projection = False
cam.position = r_OC
cam.focal_point = r_OF
cam.view_up = -e_y_cam
cam.clipping_range = (0.01, 1)
cam.Zoom(zoom)

plotter.show()

############
## solver ##
############

# control_dt = 0.04 #25 Hz
# n_load_steps = int(total_t / control_dt)
# n_load_steps = 100
controlled_steps = 100
n_load_steps = settle_steps + controlled_steps
total_t = 60.0 # s
time_scale = total_t * n_load_steps / controlled_steps
solver = Newton(system, n_load_steps=n_load_steps, options=SolverOptions(newton_atol = 1e-10, newton_rtol = 1e-6, newton_max_iter=100))
# solver.load_steps = np.linspace(0, total_t, n_load_steps + 1)
# solver.nt = len(solver.load_steps)

controller.attach_solver(solver)
sol = solver.solve()

########
# plot #
########
from matplotlib import pyplot as plt

history = controller.history
t = np.asarray(history["t"]) * time_scale # scale back to seconds in real time
u = np.asarray(history["u"]) * 1e2 # cm
uref = np.asarray(history["u_ref"]) * 1e2 # cm

fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
for i, label in enumerate(["X", "Y", "Z"]):
    ax[i].plot(t, u[:, i], label="Actual")
    ax[i].plot(t, uref[:, i], label="Desired", linestyle="--")
    ax[i].set_ylabel(f"{label} [cm]"); ax[i].grid(True); ax[i].legend()
ax[2].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()

# t = sol.t
# q_nodes = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
# plt.plot(q_nodes[:, -1, 0], q_nodes[:, -1, 2])
# plt.xlabel("x [m]")
# plt.ylabel("z [m]")
# plt.grid()
# plt.show(block=False)

plotter.render_solution(sol, True, play_speed_up=.1)