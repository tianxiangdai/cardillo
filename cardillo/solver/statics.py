import numpy as np
from tqdm import tqdm

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.math.fsolve import fsolve
from cardillo.solver.solver_options import SolverOptions
from cardillo.solver.solution import Solution


class Newton:
    """Force and displacement controlled Newton-Raphson method. This solver
    is used to find a static solution for a mechanical system. Forces and
    bilateral constraint functions are incremented in each load step if they
    depend on the time t in [0, 1]. Thus, a force controlled Newton-Raphson method
    is obtained by constructing a time constant constraint function function.
    On the other hand a displacement controlled Newton-Raphson method is
    obtained by passing constant forces and time dependent constraint functions.
    """

    def __init__(
        self,
        system,
        n_load_steps=1,
        verbose=True,
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options
        self.verbose = verbose
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = len(self.load_steps)

        self.len_t = len(str(self.nt))
        self.len_maxIter = len(str(self.options.newton_max_iter))

        # other dimensions
        self.nq = system.nq
        self.nu = system.nu

        self.split_f = np.cumsum(
            np.array(
                [system.nu, system.nla_g, system.nla_c, system.nla_S],
                dtype=int,
            )
        )
        self.split_x = np.cumsum(
            np.array(
                [system.nq, system.nla_g, system.nla_c],
                dtype=int,
            )
        )

        # initial conditions
        x0 = np.concatenate((system.q0, system.la_g0, system.la_c0))
        self.nx = len(x0)
        self.u0 = np.zeros(system.nu)  # zero velocities as system is static

        # memory allocation
        self.x = np.zeros((self.nt, self.nx), dtype=float)
        self.x[0] = x0
        self._W_g_coo = self._W_c_coo = self._h_q_coo = self._Wla_g_q_coo = (
            self._Wla_c_q_coo
        ) = self._c_q_coo = self._g_q_coo = self._g_S_q_coo = None
        self._jac_coo = CooMatrix((self.nx, self.nx))

    def fun(self, x, t):
        c0, c1, c2 = self.split_x
        r0, r1, r2, r3 = self.split_f
        # unpack unknowns
        q, la_g, la_c = x[:c0], x[c0:c1], x[c1:]

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        # csr is used for efficient matrix vector multiplication, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        self._W_g_coo = self.system.W_g(t, q, format="Coo", coo=self._W_g_coo)
        self._W_c_coo = self.system.W_c(t, q, format="Coo", coo=self._W_c_coo)
        self.W_g = self._W_g_coo.asformat("coo")
        self.W_c = self._W_c_coo.asformat("coo")

        # static equilibrium
        F = np.zeros_like(x)
        F[:r0] = self.system.h(t, q, self.u0) + self.W_g @ la_g + self.W_c @ la_c
        F[r0:r1] = self.system.g(t, q)
        F[r1:r2] = self.system.c(t, q, self.u0, la_c)
        F[r2:r3] = self.system.g_S(t, q)
        return F

    def jac(self, x, t):
        c0, c1, c2 = self.split_x
        r0, r1, r2, r3 = self.split_f
        # unpack unknowns
        q, la_g, la_c = x[:c0], x[c0:c1], x[c1:]

        jac = self._jac_coo
        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        self._h_q_coo = self.system.h_q(t, q, self.u0, format="Coo", coo=self._h_q_coo)
        self._Wla_g_q_coo = self.system.Wla_g_q(
            t, q, la_g, format="Coo", coo=self._Wla_g_q_coo
        )
        self._Wla_c_q_coo = self.system.Wla_c_q(
            t, q, la_c, format="Coo", coo=self._Wla_c_q_coo
        )
        self._c_q_coo = self.system.c_q(
            t, q, self.u0, la_c, format="Coo", coo=self._c_q_coo
        )
        self._g_q_coo = self.system.g_q(t, q, format="Coo", coo=self._g_q_coo)
        self._g_S_q_coo = self.system.g_S_q(t, q, format="Coo", coo=self._g_S_q_coo)
        c_la_c = self.system.c_la_c()

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        jac["h_q", :r0, :c0] = self._h_q_coo
        jac["Wla_g_q", :r0, :c0] = self._Wla_g_q_coo
        jac["Wla_c_q", :r0, :c0] = self._Wla_c_q_coo

        jac["W_g", :r0, c0:c1] = self.W_g
        jac["W_c", :r0, c1:c2] = self.W_c
        jac["g_q", r0:r1, :c0] = self._g_q_coo
        jac["c_q", r1:r2, :c0] = self._c_q_coo
        jac["c_la_c", r1:r2, c1:c2] = c_la_c
        jac["W_c", r1:r2, c1:c2] = self.W_c
        jac["g_S_q", r2:r3, :c0] = self._g_S_q_coo
        return jac.asformat("coo").asformat("csc")
        # return bmat([[      K, self.W_g, self.W_c,   self.W_N],
        #              [    g_q,     None,     None,       None],
        #              [    c_q,     None,   c_la_c,       None],
        #              [  g_S_q,     None,     None,       None],], format="csc")

    def __pbar_text(self, force_iter, newton_iter, error):
        return (
            f" force iter {force_iter+1:>{self.len_t}d}/{self.nt};"
            f" Newton steps {newton_iter+1:>{self.len_maxIter}d}/{self.options.newton_max_iter};"
            f" error {error:.4e}"
        )

    def solve(self):
        pbar = range(0, self.nt)
        if self.verbose:
            pbar = tqdm(pbar, leave=True)
        for i in pbar:
            sol = fsolve(
                self.fun,
                self.x[i],
                jac=self.jac,
                fun_args=(self.load_steps[i],),
                jac_args=(self.load_steps[i],),
                options=self.options,
            )
            self.x[i] = sol.x
            if self.verbose:
                pbar.set_description(self.__pbar_text(i, sol.nit, sol.error))

            if not sol.success and not self.options.continue_with_unconverged:
                # return solution up to this iteration
                if self.verbose:
                    pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{self.len_t}d}/{self.nt}"
                )
                return Solution(
                    system=self.system,
                    t=self.load_steps[: i + 1],
                    q=self.x[: i + 1, : self.split_x[0]],
                    u=np.zeros((i + 1, self.nu)),
                    la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
                    la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
                )

            # solver step callback
            self.x[i, : self.split_x[0]], _ = self.system.step_callback(
                self.load_steps[i], self.x[i, : self.split_x[0]], self.u0
            )

            # warm start for next step; store solution as new initial guess
            if i < self.nt - 1:
                self.x[i + 1] = self.x[i]

        # return solution object
        if self.verbose:
            pbar.close()
        return Solution(
            self.system,
            t=self.load_steps,
            q=self.x[: i + 1, : self.split_x[0]],
            u=np.zeros((len(self.load_steps), self.nu)),
            la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
            la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
        )
