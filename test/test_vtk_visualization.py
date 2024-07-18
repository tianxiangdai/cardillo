import numpy as np
from pathlib import Path
import trimesh
from cardillo.rods import (
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)

from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, SolverOptions, Rattle
from cardillo.forces import Force, B_Moment
from cardillo.math import e2, e3
from cardillo.discrete import RigidBody, Meshed
from cardillo.visualization import Renderer
from cardillo import System


""" Cantilever beam example from

Harsch, J., Capobianco, G. and Eugster, S. R., "Finite element formulations for constrained spatial nonlinear beam theories", 2021.
https://doi.org/10.1177/10812865211000790

4.1 Elliptic integral solutions of Euler's elastica

It is a test for the static analysis of all different rod formulations in their different versions: 
(constrained) displacement-based and (constrained) mixed formulations.
"""


def cantilever(
    Rod,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=3,
    reduced_integration=True,
    constitutive_law=Harsch2021,
    title="set_a_plot_title",
):
    print(title)
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    # cross_section = RectangularCrossSection(width, width)
    cross_section = CircularCrossSection(width)

    # material properties
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])
    material_model = constitutive_law(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )
    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    clamping_left = RigidConnection(system.origin, cantilever, xi2=(0,))

    # assemble the system
    system.add(cantilever)
    system.add(clamping_left)

    # spatially fixed load at cantilever tip
    P = lambda t: material_model.Fi[2] * (10 * t) / length**2
    F = lambda t: -P(t) * e2
    force = Force(F, cantilever, (1,))
    system.add(force)

    # moment at cantilever tip
    M = lambda t: 2.5 * P(t) * e3
    moment = B_Moment(M, cantilever, (1,))
    system.add(moment)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ren = Renderer(system)
    ren.start_step_render(sync=False)

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1.0e-8),
    )
    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    ren.stop_step_render()
    ren.render_solution(sol, True)


def dzhanibekov_effect():
    ############
    # parameters
    ############

    # initial conditions
    r_OC0 = np.zeros(3)
    v_C0 = np.zeros(3)
    phi_dot0 = 20
    B_Omega_disturbance = np.array((1e-10, 0, 0))  # disturbance is required
    B_Omega0 = np.array((0, 0, phi_dot0)) + B_Omega_disturbance

    # simulation parameters
    t1 = 10  # final time

    # initialize system
    system = System()

    #################
    # assemble system
    #################

    # load mesh using trimesh
    dir_name = Path(__file__).parent.parent
    path = Path(dir_name, "examples", "dzhanibekov_effect", "mesh", "screwdriver.stl")
    mesh = trimesh.load_mesh(path)
    scale = 1 / 1000  # m/mm
    mesh.apply_transform(np.diag([scale, scale, scale, 1]))

    # quantities in mesh/body-fixed basis
    B_r_PC = mesh.center_mass  # vector from mesh origin to center of mass
    mass = mesh.mass
    B_Theta_C = mesh.moment_inertia

    q0 = RigidBody.pose2q(r_OC0, np.eye(3))
    u0 = np.hstack([v_C0, B_Omega0])
    screwdriver = Meshed(RigidBody)(
        mesh_obj=path,
        scale=1e-3,
        B_r_CP=-B_r_PC,
        A_BM=np.eye(3),
        mass=mass,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
        name="screwdriver",
    )
    system.add(screwdriver)

    # assemble system
    system.assemble()

    ###################
    # initialize render
    ###################
    render = Renderer(system)
    render.start_step_render(sync=False)

    ############
    # simulation
    ############
    dt = 1e-2  # time step
    solver = Rattle(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    render.stop_step_render()
    render.render_solution(sol, repeat=True)
    render.start_interaction(sol.t[-1], sol.q[-1], sol.u[-1])


if __name__ == "__main__":
    ############################
    # Quaternion interpolation #
    ############################
    # displacement-based formulation

    dzhanibekov_effect()

    cantilever(
        Rod=make_CosseratRod(interpolation="Quaternion", mixed=False),
        n_load_steps=1000,
        constitutive_law=Harsch2021,
        title="shear-deformable (blue): D-B quaternion interpolation",
    )
