import numpy as np
from matplotlib import pyplot as plt

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection

from cardillo.solver import Newton, SolverOptions
from cardillo.forces import B_Moment

from cardillo.system import System

from cardillo.rods.discreteRod import DiscreteRod

from cProfile import Profile

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
Q = DiscreteRod.straight_configuration(nelement, L)
rod = DiscreteRod(
    cross_section,
    material_model,
    nelement,
    Q,
    cross_section_inertias=cross_section_inertias,
)
# nodes = rod.nodes

system = System()
force = B_Moment(lambda t: t * np.array([0, 0, EI / L * 2 * np.pi]), rod, xi=1)
rc = RigidConnection(system.origin, rod, xi2=0)

system.add(rod)
system.add(force)
system.add(rc)
system.assemble()

t1 = 1

options = SolverOptions(newton_atol=1e-6, newton_rtol=1e-6)
# warm up
solver = Newton(system, n_load_steps=10, options=options, verbose=False)
solver.solve()
#
solver = Newton(system, n_load_steps=100, options=options)
# prof = Profile()
# prof.enable()
sol1 = solver.solve()
# prof.disable()
# prof.dump_stats('statics1.prof')


###########
# mixed rod
###########
Rod = make_CosseratRod(polynomial_degree=1)
Q = Rod.straight_configuration(nelement, L)
rod = Rod(
    cross_section,
    material_model,
    nelement,
    Q=Q,
    cross_section_inertias=cross_section_inertias,
)

system = System()
force = B_Moment(lambda t: t * np.array([0, 0, EI / L * 2 * np.pi]), rod, xi=1)
rc = RigidConnection(rod, system.origin, xi1=0)

system.add(rod)
system.add(force)
system.add(rc)
system.assemble()

solver = Newton(system, n_load_steps=100, options=options)
prof = Profile()
# prof.enable()
sol2 = solver.solve()
# prof.disable()
# prof.dump_stats('statics2.prof')

######
# plot
######
t, q = sol1.t, sol1.q
r_OC1s = q[-1, rod.qDOF].reshape((-1, 7))[:, :3]

t, q = sol2.t, sol2.q
r_OC2s = q[-1, rod.qDOF][rod.nodalDOF_r]

print(np.linalg.norm(r_OC1s - r_OC2s, np.inf))

plt.subplot(2, 1, 1)
plt.plot(r_OC1s[:, 0], r_OC1s[:, 1], "-xr")
plt.plot(r_OC2s[:, 0], r_OC2s[:, 1], "-b.")
plt.axis("equal")
plt.subplot(2, 1, 2)
plt.plot(np.linalg.norm(r_OC1s - r_OC2s, axis=1), "-b.")
plt.yscale("log")
plt.show(block=True)
