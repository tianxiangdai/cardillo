from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import Sphere, PointMass
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.solver import BackwardEuler, SolverOptions

if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    m = 1  # mass
    l0 = 1  # undeformed length of spring
    k = 1e8  # spring stiffness
    d = 0  # damping constant

    n_particles = 20  # number of particles

    gravity = np.array([0, 0, -10])  # gravity

    #######################
    # simulation parameters
    #######################
    compliance_form = True  # use compliance formulation for force element, i.e., force is Lagrange multiplier la_c.
    t0 = 0  # initial time
    t1 = 5  # final time

    #################
    # assemble system
    #################
    system = System(t0=t0)

    
    # add particles as point masses and add gravity for each
    particles = []
    offset = np.array([l0, 0, 0])

    u0 = np.zeros(3)
    for i in range(n_particles):
        q0 = (i + 1) * offset
        particle = Sphere(PointMass)(
            radius=l0 / 20, mass=m, q0=q0, u0=u0, name="mass" + str(i)
        )
        system.add(particle)
        particles.append(particle)
        system.add(Force(m * gravity, particle, name="gravity" + str(i)))

    # spring-damper between origin and first particle of chain
    system.add(
        SpringDamper(
            TwoPointInteraction(system.origin, particles[0]),
            k,
            d,
            l_ref=l0,
            compliance_form=compliance_form,
            name="spring_damper_0",
        )
    )
    # spring-damper between subsequent particles of chain
    for i in range(n_particles - 1):
        system.add(
            SpringDamper(
                TwoPointInteraction(particles[i], particles[i + 1]),
                k,
                d,
                l_ref=l0,
                compliance_form=compliance_form,
                name="spring_damper_" + str(i + 1),
            )
        )

    system.assemble()

    ############
    # simulation
    ############
    dt = 1e-2  # time step
    solver = BackwardEuler(
        system, t1, dt, options=SolverOptions(newton_max_iter=50)
    )  # create solver
    sol = solver.solve()  # simulate system
    t = sol.t
    q = sol.q
    u = sol.u

    ############
    # VTK export
    ############
    # path = Path(__file__)
    # system.export(path.parent, "vtk", sol)

    ###########################
    # animation with matplotlib
    ###########################

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    width = 1.5 * n_particles * l0
    ax.axis("equal")
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    (line,) = ax.plot([], [], "-ok")

    def update(t, q, line):
        r = np.array([particle.r_OP(t, q[particle.qDOF]) for particle in particles])
        x = [0]
        z = [0]
        x.extend(r[:, 0])
        z.extend(r[:, 2])
        line.set_data(x, z)
        return (line,)

    def animate(i):
        update(t[i], q[i], line)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
