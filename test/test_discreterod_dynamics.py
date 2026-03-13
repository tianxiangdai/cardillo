import numpy as np
from matplotlib import pyplot as plt

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection

from cardillo.solver import *
from cardillo.forces import *

from cardillo.system import System

from cardillo.rods.discreteRod import DiscreteRod

import cProfile

nelement = 20
radius = 0.03
L = 1
density = 0.4 / (L * np.pi * radius**2)
cross_section = CircularCrossSection(radius)
cross_section_inertias = CrossSectionInertias(
    density=density, cross_section=cross_section
)

E, G = 7e5, 2e5
EA = E * cross_section.area
EI = E * cross_section.second_moment[1, 1]
GA = G * cross_section.area
GJ = G * cross_section.second_moment[0, 0]
material_model = Simo1986(
    np.array([EA, GA, GA]),
    np.array([GJ, EI, EI]),
)

# ##############
# # discrete rod
# ##############
Q1 = DiscreteRod.straight_configuration(nelement, L)
rod1 = DiscreteRod(
    cross_section,
    material_model,
    nelement,
    Q1,
    cross_section_inertias=cross_section_inertias,
)
# nodes = rod.nodes

system1 = System()
f_fun = lambda t: t * np.array([0, -0.5, 0])
force = Force(f_fun, rod1, xi=1)
rc = RigidConnection(system1.origin, rod1, xi2=0)

# system1.add(*nodes)
system1.add(rod1)
system1.add(force)
system1.add(rc)
system1.assemble()

solver = Newton(system1, n_load_steps=10)
sol_statics = solver.solve()
system1.set_new_initial_state(sol_statics.q[-1], sol_statics.u[-1])

system1.remove(force)
system1.assemble()

t1 = 3
rtol = 1e-3
atol = 1e-6
solver = ScipyDAE(system1, t1, t1 / 1000, rtol=rtol, atol=atol)

from time import perf_counter

t0 = perf_counter()
# prof = cProfile.Profile()
# prof.enable()
sol1 = solver.solve()
# prof.disable()
# prof.dump_stats("prof1.prof")
print("time:", perf_counter() - t0)
# exit()


###########
# mixed rod
###########
Rod = make_CosseratRod(polynomial_degree=1)
Q2 = Rod.straight_configuration(nelement, L)
rod2 = Rod(
    cross_section,
    material_model,
    nelement,
    Q=Q2,
    cross_section_inertias=cross_section_inertias,
)

system2 = System()

force = Force(f_fun, rod2, xi=1)
rc = RigidConnection(system2.origin, rod2, xi2=0)


system2.add(rod2)
system2.add(force)
system2.add(rc)
system2.assemble()


def cvt(src, n):
    if src.ndim == 1:
        return src.reshape((-1, n)).T.flatten()
    src = np.swapaxes(src.reshape((src.shape[0], -1, n)), 1, 2).reshape(
        (src.shape[0], -1)
    )
    src = np.swapaxes(src.reshape((-1, n, src.shape[1])), 0, 1).reshape(
        (-1, src.shape[1])
    )
    return src


system2.set_new_initial_state(cvt(sol_statics.q[-1], 7), cvt(sol_statics.u[-1], 6))

system2.remove(force)
system2.assemble()

solver = ScipyDAE(system2, t1, t1 / 1000, rtol=rtol, atol=atol)

# solver = Newton(system, n_load_steps=10)
# from cardillo.visualization import Renderer
# ren = Renderer(system2, [rod])
# ren.start_step_render()

t0 = perf_counter()
# prof = cProfile.Profile()
# prof.enable()
sol2 = solver.solve()
# prof.disable()
# prof.dump_stats("prof2.prof")
# exit()
print("time:", perf_counter() - t0)

# print(np.allclose(r_OC1s, r_OC2s))


# r_OC1s = sol1.q[-1, rod.qDOF][rod.nodalDOF_r]
# r_OC2s = sol2.q[-1, rod.qDOF][rod.nodalDOF_r]

# plt.plot(r_OC1s[:, 0], r_OC1s[:, 1], "-xr")
# plt.plot(r_OC2s[:, 0], r_OC2s[:, 1], "-b.")
# plt.axis("equal")
# plt.show(block=True)

"""

def cvt(M, n):
    if M.ndim == 1:
        return M.reshape((-1, n)).T.flatten()
    M = np.swapaxes(M.reshape((M.shape[0], -1, n)), 1, 2).reshape((M.shape[0], -1))
    M = np.swapaxes(M.reshape((-1, n, M.shape[1])), 0, 1).reshape((-1, M.shape[1]))
    return M
"""
"""
compare = lambda A, B: print(np.linalg.norm(A - B, np.inf))
q1 = sol1.q[0]
q2 = sol2.q[0]
u1 = sol1.u[0]
u2 = sol2.u[0]
la_c1 = sol1.la_c[0]
la_c2 = sol2.la_c[0]
la_g1 = sol1.la_g[0]
la_g2 = sol2.la_g[0]


compare(cvt(Q1, 7), Q2)
compare(cvt(q1, 7), q2)
compare(cvt(u1, 7), u2)
compare(cvt(la_c1, 6), la_c2)
compare(la_g1, la_g2)

"""
"""
t = sol1.t[-1]
q1 = sol1.q[-1]
u1 = sol1.u[-1]
la_c1 = sol1.la_c[-1]
q2 = q1.reshape((-1,7)).T.flatten()
u2 = u1.reshape((-1,6)).T.flatten()
la_c2 = la_c1.reshape((-1,6)).T.flatten()


# M
m1 = system1.M(t, q1).toarray().diagonal()
m2 = system2.M(t, q2).toarray().diagonal()
assert np.allclose(m1.reshape((-1,6)).T.flatten(), m2)

# c_la_c
c_la_c1 = system1.c_la_c().toarray().diagonal()
c_la_c2 = system2.c_la_c().toarray().diagonal()
assert np.allclose(c_la_c1.reshape((-1,6)).T.flatten(), c_la_c2)

# W_c
W_c1 = system1.W_c(t, q1).toarray()
W_c1 = np.swapaxes(W_c1.reshape((W_c1.shape[0], -1, 6)), 1, 2).reshape((W_c1.shape[0], -1))
W_c1 = np.swapaxes(W_c1.reshape((-1, 6, W_c1.shape[1])), 0, 1).reshape((-1, W_c1.shape[1]))
W_c2 = system2.W_c(t, q2).toarray()
assert np.allclose(W_c1, W_c2)

c1 = system1.c(t, q1, u1, la_c1)
c2 = system2.c(t, q2, u2, la_c2)
assert np.allclose(c1.reshape((-1,6)).T.flatten(), c2)

# h
h1 = system1.h(t, q1, u1)
h2 = system2.h(t, q2, u2)
assert np.allclose(h1.reshape((-1,6)).T.flatten(), h2)
"""

# plot result
plt.subplot(2, 2, 1)
plt.plot(sol1.t, sol1.q[:, rod1.qDOF][:, rod1.nodalDOF_r[-1]][:, 0])
plt.plot(sol2.t, sol2.q[:, rod2.qDOF][:, rod2.nodalDOF_r[-1][0]])
plt.grid()
plt.subplot(2, 2, 2)
plt.plot(
    sol1.t,
    sol1.q[:, rod1.qDOF][:, rod1.nodalDOF_r[-1]][:, 0]
    - sol2.q[:, rod2.qDOF][:, rod2.nodalDOF_r[-1][0]],
)
plt.yscale("log")
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(sol1.t, sol1.q[:, rod1.qDOF][:, rod1.nodalDOF_r[-1]][:, 1])
plt.plot(sol2.t, sol2.q[:, rod2.qDOF][:, rod2.nodalDOF_r[-1][1]])
plt.grid()
plt.subplot(2, 2, 4)
plt.plot(
    sol1.t,
    np.abs(
        sol1.q[:, rod1.qDOF][:, rod1.nodalDOF_r[-1]][:, 1]
        - sol2.q[:, rod2.qDOF][:, rod2.nodalDOF_r[-1][1]]
    ),
)
plt.yscale("log")
plt.grid()
plt.show(block=True)

# # render solution
# step = int(len(q)//1000)
# sol = Solution(system, t_eval[::step], q[::step], u[::step])
# ren = Renderer(system)
# ren.render_solution(sol, repeat=True)
