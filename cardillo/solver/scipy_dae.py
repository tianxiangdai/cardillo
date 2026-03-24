import numpy as np
from scipy.sparse import eye_array, lil_array
from scipy_dae.integrate import solve_dae
from tqdm import tqdm

from cardillo.solver import Solution, SolverSummary
from cardillo.utility.coo_matrix import CooMatrix


# TODO:
# - Add Jacobian of GGl term if convergence problems occur
class ScipyDAE:
    """Wrapper around Radau IIA and BDF methods implementted in `scipy_dae`. 
    A stabilized index 1 formulation is used as proposed by Anantharaman and Hiller.

    References:
    -----------
    scipy_dae: https://github.com/JonasBreuling/scipy_dae \\
    Anantharaman and Hiller.: https://doi.org/10.1002/nme.1620320803
    """

    def __init__(
        self,
        system,
        t1,
        dt,
        method="Radau",
        rtol=1.0e-3,
        atol=1.0e-6,
        **kwargs,
    ):
        self.system = system
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.kwargs = kwargs

        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_c = self.system.nla_c
        self.ny = self.nq + self.nu + 2 * self.nla_g + self.nla_gamma + self.nla_c
        self.split = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                ],
                dtype=int,
            )
        )[:-1]
        self.y0 = np.concatenate(
            (
                system.q0,
                system.u0,
                0 * system.la_g0,
                0 * system.la_g0,
                0 * system.la_gamma0,
                0 * system.la_c0,
            )
        )
        self.y_dot0 = np.concatenate(
            (
                system.q_dot0,
                system.u_dot0,
                0 * system.la_g0,  # GGL multiplier
                system.la_g0,
                system.la_gamma0,
                system.la_c0,
            )
        )

        # integration time
        self.t0 = system.t0
        self.t1 = (
            t1
            if t1 > self.t0
            else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t_eval = np.arange(self.t0, self.t1 + self.dt, self.dt)

        self.frac = (t1 - self.t0) / 101
        self.pbar = tqdm(total=100, leave=True)
        self.i = 0
        self._init_coo()
        # residual
        self._F = np.zeros(self.ny, dtype=float)

    def _init_coo(self):
        t = self.t0
        # unpack vectors
        q, u, _, _, _, _ = np.array_split(self.y0, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(
            self.y_dot0, self.split
        )

        # evaluate commonly used quantities
        q_dot_q = self.system.q_dot_q(t, q, u)
        q_dot_u = self.system.q_dot_u(t, q)

        Mu_q = self.system.Mu_q(t, q, u_dot, "CooMatrix")
        h_q = self.system.h_q(t, q, u, "CooMatrix")
        h_u = self.system.h_u(t, q, u, "CooMatrix")
        Wla_tau_q = self.system.Wla_tau_q(t, q, u, "CooMatrix")
        Wla_tau_u = self.system.Wla_tau_u(t, q, u, "CooMatrix")
        Wla_g_q = self.system.Wla_g_q(t, q, la_g, "CooMatrix")
        Wla_gamma_q = self.system.Wla_gamma_q(t, q, la_gamma, "CooMatrix")
        Wla_c_q = self.system.Wla_c_q(t, q, la_c, "CooMatrix")

        g_dot_q = self.system.g_dot_q(t, q, u, "CooMatrix")
        g_dot_u = self.system.g_dot_u(t, q, "CooMatrix")

        gamma_q = self.system.gamma_q(t, q, u, "CooMatrix")
        gamma_u = self.system.gamma_u(t, q, "CooMatrix")

        c_q = self.system.c_q(t, q, u, la_c, "CooMatrix")
        c_u = self.system.c_u(t, q, u, la_c, "CooMatrix")

        eye_q = eye_array(self.nq)
        M = self.system.M(t, q)
        g_q = self.system.g_q(t, q, "CooMatrix")
        W_g = self.system.W_g(t, q, "CooMatrix")
        W_gamma = self.system.W_gamma(t, q, "CooMatrix")
        W_c = self.system.W_c(t, q, "CooMatrix")
        c_la_c = self.system.c_la_c()

        # first Jacobian w.r.t. y
        self._Jy_coo = CooMatrix((self.ny, self.ny))
        self._Jy_coo.allocate_data(
            np.arange(self.split[0]), np.arange(self.split[0]), -q_dot_q
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0]), np.arange(self.split[0], self.split[1]), -q_dot_u
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0]),
            Mu_q,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0]),
            -h_q,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0]),
            -Wla_tau_q,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0]),
            -Wla_gamma_q,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]), np.arange(self.split[0]), -Wla_g_q
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0]),
            -Wla_c_q,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0], self.split[1]),
            -h_u,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0], self.split[1]),
            -Wla_tau_u,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[1], self.split[2]), np.arange(self.split[0]), g_q
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[2], self.split[3]), np.arange(self.split[0]), g_dot_q
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[2], self.split[3]),
            np.arange(self.split[0], self.split[1]),
            g_dot_u,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[3], self.split[4]), np.arange(self.split[0]), gamma_q
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[3], self.split[4]),
            np.arange(self.split[0], self.split[1]),
            gamma_u,
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[4], self.ny), np.arange(self.split[0]), c_q
        )
        self._Jy_coo.allocate_data(
            np.arange(self.split[4], self.ny),
            np.arange(self.split[0], self.split[1]),
            c_u,
        )
        self._Jy_coo.fix_size()

        self._Jyp_coo = CooMatrix((self.ny, self.ny))
        self._Jyp_coo.allocate_data(
            np.arange(self.split[0]), np.arange(self.split[0]), eye_q
        )
        self._Jyp_coo.allocate_data(
            np.arange(self.split[0]), np.arange(self.split[1], self.split[2]), -g_q.T
        )
        self._Jyp_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[0], self.split[1]),
            M,
        )
        self._Jyp_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[2], self.split[3]),
            -W_g,
        )
        self._Jyp_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[3], self.split[4]),
            -W_gamma,
        )
        self._Jyp_coo.allocate_data(
            np.arange(self.split[0], self.split[1]),
            np.arange(self.split[4], self.ny),
            -W_c,
        )
        self._Jyp_coo.allocate_data(
            np.arange(self.split[4], self.ny), np.arange(self.split[4], self.ny), c_la_c
        )
        self._Jyp_coo.fix_size()
        self._Jyp_coo.set_allocated_data(0, eye_q)
        self._Jyp_coo.set_allocated_data(6, c_la_c)

    def event(self, t, y, yp):
        i0, i1 = self.split[:2]
        q = y[:i0]
        u = y[i0:i1]
        q, u = self.system.step_callback(t, q, u)
        return 1

    def fun(self, t, y, yp):
        # update progress bar
        i1 = int(t // self.frac)
        self.pbar.set_description(f"t: {t:0.2e}s < {self.t1:0.2e}s", refresh=False)
        self.pbar.update(i1 - self.i)
        self.i = i1

        # unpack vectors
        i0, i1, i2, i3, i4 = self.split[:5]
        q = y[:i0]
        u = y[i0:i1]

        q_dot = yp[:i0]
        u_dot = yp[i0:i1]
        mu_g = yp[i1:i2]
        la_g = yp[i2:i3]
        la_gamma = yp[i3:i4]
        la_c = yp[i4:]

        system = self.system

        ####################
        # kinematic equation
        ####################
        q_dot = q_dot - system.q_dot(t, q, u)
        if mu_g.size > 0:
            q_dot -= mu_g @ system.g_q(t, q)
        self._F[:i0] = q_dot

        #####################
        # equations of motion
        #####################
        h = system.M(t, q) @ u_dot
        h -= system.h(t, q, u)
        la_tau = system.la_tau(t, q, u)
        if la_tau.size > 0:
            h -= system.W_tau(t, q) @ la_tau
        if la_g.size > 0:
            h -= system.W_g(t, q) @ la_g
        if la_gamma.size > 0:
            h -= system.W_gamma(t, q) @ la_gamma
        if la_c.size > 0:
            h -= system.W_c(t, q) @ la_c
        self._F[i0:i1] = h

        #######################
        # bilateral constraints
        #######################
        if i2 > i1:
            self._F[i1:i2] = system.g(t, q)
        if i3 > i2:
            self._F[i2:i3] = system.g_dot(t, q, u)
        if i4 > i3:
            self._F[i3:i4] = system.gamma(t, q, u)

        ############
        # compliance
        ############
        if y.size > i4:
            self._F[i4:] = system.c(t, q, u, la_c)

        return self._F

    def jac(self, t, y, yp):
        # unpack vectors
        i0, i1, i2, i3, i4 = self.split[:5]
        q = y[:i0]
        u = y[i0:i1]

        u_dot = yp[i0:i1]
        la_g = yp[i2:i3]
        la_gamma = yp[i3:i4]
        la_c = yp[i4:]

        # evaluate commonly used quantities
        system = self.system

        # first Jacobian w.r.t. y
        Jy_coo = self._Jy_coo
        if Jy_coo.data_allocation_length(0):
            q_dot_q = system.q_dot_q(t, q, u)
            Jy_coo.set_allocated_data(0, -q_dot_q)
        if Jy_coo.data_allocation_length(1):
            q_dot_u = system.q_dot_u(t, q)
            Jy_coo.set_allocated_data(1, -q_dot_u)
        if Jy_coo.data_allocation_length(2):
            Mu_q = system.Mu_q(t, q, u_dot, "CooMatrix")
            Jy_coo.set_allocated_data(2, Mu_q)
        if Jy_coo.data_allocation_length(3):
            h_q = system.h_q(t, q, u, "CooMatrix")
            Jy_coo.set_allocated_data(3, -h_q)
        if Jy_coo.data_allocation_length(4):
            Wla_tau_q = system.Wla_tau_q(t, q, u, "CooMatrix")
            Jy_coo.set_allocated_data(4, -Wla_tau_q)
        if Jy_coo.data_allocation_length(5):
            Wla_gamma_q = system.Wla_gamma_q(t, q, la_gamma, "CooMatrix")
            Jy_coo.set_allocated_data(5, -Wla_gamma_q)
        if Jy_coo.data_allocation_length(6):
            Wla_g_q = system.Wla_g_q(t, q, la_g, "CooMatrix")
            Jy_coo.set_allocated_data(6, -Wla_g_q)
        if Jy_coo.data_allocation_length(7):
            Wla_c_q = system.Wla_c_q(t, q, la_c, "CooMatrix")
            Jy_coo.set_allocated_data(7, -Wla_c_q)
        if Jy_coo.data_allocation_length(8):
            h_u = system.h_u(t, q, u, "CooMatrix")
            Jy_coo.set_allocated_data(8, -h_u)
        if Jy_coo.data_allocation_length(9):
            Wla_tau_u = system.Wla_tau_u(t, q, u, "CooMatrix")
            Jy_coo.set_allocated_data(9, -Wla_tau_u)
        if Jy_coo.data_allocation_length(10):
            g_q = system.g_q(t, q, "CooMatrix")
            Jy_coo.set_allocated_data(10, g_q)
        if Jy_coo.data_allocation_length(11):
            g_dot_q = system.g_dot_q(t, q, u, "CooMatrix")
            Jy_coo.set_allocated_data(11, g_dot_q)
        if Jy_coo.data_allocation_length(12):
            g_dot_u = system.g_dot_u(t, q, "CooMatrix")
            Jy_coo.set_allocated_data(12, g_dot_u)
        if Jy_coo.data_allocation_length(13):
            gamma_q = system.gamma_q(t, q, u, "CooMatrix")
            Jy_coo.set_allocated_data(13, gamma_q)
        if Jy_coo.data_allocation_length(14):
            gamma_u = system.gamma_u(t, q, "CooMatrix")
            Jy_coo.set_allocated_data(14, gamma_u)
        if Jy_coo.data_allocation_length(15):
            c_q = system.c_q(t, q, u, la_c, "CooMatrix")
            Jy_coo.set_allocated_data(15, c_q)
        if Jy_coo.data_allocation_length(16):
            c_u = system.c_u(t, q, u, la_c, "CooMatrix")
            Jy_coo.set_allocated_data(16, c_u)

        # second Jacobian w.r.t. yp
        Jyp_coo = self._Jyp_coo
        # eye_q = eye_array(self.nq)
        # Jyp_coo.set_allocated(0, eye_q)
        if Jyp_coo.data_allocation_length(1):
            Jyp_coo.set_allocated_data(1, -g_q.T)
        if Jyp_coo.data_allocation_length(2):
            M = system.M(t, q)
            Jyp_coo.set_allocated_data(2, M)
        if Jyp_coo.data_allocation_length(3):
            W_g = system.W_g(t, q, "CooMatrix")
            Jyp_coo.set_allocated_data(3, -W_g)
        if Jyp_coo.data_allocation_length(4):
            W_gamma = system.W_gamma(t, q, "CooMatrix")
            Jyp_coo.set_allocated_data(4, -W_gamma)
        if Jyp_coo.data_allocation_length(5):
            W_c = system.W_c(t, q, "CooMatrix")
            Jyp_coo.set_allocated_data(5, -W_c)
        # c_la_c = system.c_la_c()
        # Jyp_coo.set_allocated(6, c_la_c)

        return Jy_coo.asformat("coo"), Jyp_coo.asformat("coo")

        # note: Keep this for debugging the Jacobian

        # from scipy.optimize._numdiff import approx_derivative

        # Jy_num = approx_derivative(lambda y: self.fun(t, y, yp), y, method="2-point")
        # diff_Jy = Jy - Jy_num
        # diff_Jy = diff_Jy[self.split[0]:, self.split[0]:] # ignore kinematic equations since GGL Jacobian use not implemented
        # error_Jy = np.linalg.norm(diff_Jy)
        # print(f"error_Jy: {error_Jy}")

        # Jyp_num = approx_derivative(lambda yp: self.fun(t, y, yp), yp, method="2-point")
        # diff_Jyp = Jyp - Jyp_num
        # error_Jyp = np.linalg.norm(diff_Jyp)
        # print(f"error_Jyp: {error_Jyp}")

        # return Jy_num, Jyp_num

    def solve(self):
        solver_summary = SolverSummary(f"Scipy solve_dae with method '{self.method}'")
        sol = solve_dae(
            self.fun,
            self.t_eval[[0, -1]],
            self.y0,
            self.y_dot0,
            t_eval=self.t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            events=[self.event],
            jac=self.jac,
            **self.kwargs,
        )
        self.pbar.close()
        # solver_summary.print()

        # unpack solution
        t = sol.t
        q, u, _, _, _, _ = np.array_split(sol.y, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(sol.yp, self.split)

        return Solution(
            system=self.system,
            t=t,
            q=q.T,
            u=u.T,
            q_dot=q_dot.T,
            u_dot=u_dot.T,
            mu_g=mu_g.T,
            la_g=la_g.T,
            la_gamma=la_gamma.T,
            la_c=la_c.T,
            solver_summary=solver_summary,
        )
