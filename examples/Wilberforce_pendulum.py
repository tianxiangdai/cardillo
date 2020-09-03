from cardillo.model.classical_beams.spatial import Hooke_quadratic
from cardillo.model.classical_beams.spatial import Timoshenko_director_dirac
from cardillo.model.classical_beams.spatial import Timoshenko_director_integral, Euler_Bernoulli_director_integral, Inextensible_Euler_Bernoulli_director_integral
from cardillo.discretization.B_spline import fit_B_Spline
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.solver import Euler_backward_singular, Generalized_alpha_4_singular_index1, Generalized_alpha_4_singular_index3
from cardillo.model.force import Force
from cardillo.model.line_force.line_force import Line_force
from cardillo.math.algebra import e3, ax2skew
from cardillo.model.rigid_body import Rigid_body_euler
from cardillo.solver.solution import load_solution

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

c = 8.0e-3 # eccentricity

# TODO: this inertia tensor is not w.r.t. the center of mass! (maybe set screw centered in this example)
def cylinder():
    rho = 7850 # [kg / m^3]; steel
    R = 18e-3 # radius
    # L = 50e-3 # height
    L = 49e-3 # height; fixed in order to get same weight as measured
    e = 10e-3 # eccentricity
    V = np.pi * R**2 * L # volume
    m = V * rho # mass

    I11 = I22 = (1/4) * m * R**2 + (1/12) * m * L**2
    I33 = 0.5 * m * R**2
    Theta_S = np.diag([I11, I22, I33]) # intertia tensor
    r = np.array([0, 0, -e])
    r_tilde = ax2skew(r)
    Theta_Steiner = m * r_tilde.T @ r_tilde # Steiner part
    Theta = Theta_S + Theta_Steiner

    return m, Theta

def disc(a, coil_radius):
    rho = 7850 # [kg / m^3]; steel
    # rho = 2698.9 # [kg / m^3]; aluminum
    b = 5.925e-3 # height
    # # c = 3.0e-3 # eccentricity (best results for n=20, nEl=128)
    # c = 4.0e-3 # eccentricity
    V = np.pi * (a / 2)**2 * b # volume
    m = V * rho # mass inner disc

    I11 = 0.5 * m * (a/2)**2
    I22 = I33 = (1/4) * m * (a/2)**2 + (1/12) * m * b**2
    Theta_S13 = np.diag([I11, I22, I33]) # intertia tensor
    Theta_S24 = np.diag([I22, I11, I33]) # intertia tensor ( rotated with pi/2 around d3)
    r_13 = np.array([coil_radius + c, 0, 0])
    r_24 = np.array([0, coil_radius + c, 0])
    r_13_tilde = ax2skew(r_13)
    r_24_tilde = ax2skew(r_24)
    Theta_Steiner = m * ( 2 * r_13_tilde.T @ r_13_tilde + 2 * r_24_tilde.T @ r_24_tilde ) # Steiner part
    Theta = 2 * Theta_S13 + 2 * Theta_S24 + Theta_Steiner
    return 4 * m, Theta

def screw(coil_radius):
    rho = 7850 # [kg / m^3]; steel
    R = 18e-3 # radius central cylinder
    d = 5.01e-3
    l = 33e-3
    V = np.pi * (d/2)**2 * l # volume
    m = V * rho # mass

    I11 = 0.5 * m * (d/2)**2
    I22 = I33 = (1/4) * m * (d/2)**2 + (1/12) * m * l**2
    Theta_S13 = np.diag([I11, I22, I33]) # intertia tensor
    Theta_S24 = np.diag([I22, I11, I33]) # intertia tensor ( rotated with pi/2 around d3)
    r_13 = np.array([coil_radius + l/2, 0, 0])
    r_24 = np.array([0, coil_radius + l/2, 0])
    r_13_tilde = ax2skew(r_13)
    r_24_tilde = ax2skew(r_24)
    Theta_Steiner = m * ( 2 * r_13_tilde.T @ r_13_tilde + 2 * r_24_tilde.T @ r_24_tilde ) # Steiner part
    Theta = 2 * Theta_S13 + 2 * Theta_S24 + Theta_Steiner
    return 4 * m, Theta

def Wilberforce_bob(coil_radius, debug=True):
    ##########
    # cylinder
    ##########
    m1, Theta1 = cylinder()
    if debug:
        print(f'mass cylinder = {m1};')
        # print(f'inertia cylinder =\n{Theta1}')

    ###########
    # 4 screw's
    ###########
    m2, Theta2 = screw(coil_radius)
    if debug:
        print(f"mass 4 screw's = {m2}")
        # print(f"inertia 4 screw's  =\n{Theta2}")

    # compute total mass of cylinder and screw's
    # this can be measured
    if debug:
        print(f"mass cylinder & 4 screw's = {m1 + m2}; measured masses = {0.412}; error = {np.abs(m1 + m2 - 0.412) / 0.412}")
        # print(f"inertia cylinder & 4 screw's  =\n{Theta1 + Theta2}")

    ##########
    # 4 disc's
    ##########
    a1 = 5e-3 # radius
    a2 = 19e-3 # radius
    m3_1, Theta3_1 = disc(a1, coil_radius)
    m3_2, Theta3_2 = disc(a2, coil_radius)
    m3 = m3_2 - m3_1
    Theta3 = Theta3_2 - Theta3_1
    if debug:
        print(f"mass 4 disc's = {m3}; measured mass = {0.049}; error = {np.abs(m3 - 0.049) / 0.049}")
        # print(f"inertia 4 disc's =\n{Theta3}")

    m = m1 + m2 + m3
    Theta = Theta1 + Theta2 + Theta3

    if debug:
        print(f'total mass = {m}')
        print(f'total inertia =\n{Theta}')

    return m, Theta

def helix3D(t, r, c, plane='xyz'):
    """Helix function in xyz plane, see https://mathworld.wolfram.com/Helix.html
    """
    assert len(plane) == 3
    orientation = 'xyz'
    perm = []
    for i in plane:
        perm.append(orientation.find(i))

    nt = len(t)
    P = np.zeros((nt, 3))
    dP = np.zeros((nt, 3))
    ddP = np.zeros((nt, 3))

    pi2 = 2 * np.pi
    rc_phi = r * np.cos(pi2 * t)
    rs_phi = r * np.sin(pi2 * t)
        
    P[:, perm] = np.array([rc_phi, rs_phi, c * t]).T
    dP[:, perm] = np.array([-pi2 * rs_phi, pi2 * rc_phi, c * np.ones_like(t)]).T
    ddP[:, perm] = np.array([-pi2**2*rc_phi, -pi2**2*rs_phi, np.zeros_like(t)]).T

    return P, dP, ddP

# Beam = Timoshenko_director_dirac
Beam = Timoshenko_director_integral
# Beam = Euler_Bernoulli_director_integral
# Beam = Inextensible_Euler_Bernoulli_director_integral

# statics = True
statics = False

save = True
# save = False

import os
path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ################################################################################################
    # beam parameters, taken from https://faraday.physics.utoronto.ca/PHY182S/WilberforceRefBerg.pdf
    ################################################################################################
    # see https://de.wikipedia.org/wiki/Fl%C3%A4chentr%C3%A4gheitsmoment#Beispiele
    d = 1e-3 # 1mm cross sectional diameter
    r = d / 2  # cross sectional radius
    A = np.pi * r**2
    I2 = I3 = (np.pi / 4) * r**4
    Ip = I2 + I3
    
    # Federstahl nach EN 10270-1
    rho = 7850 # [kg / m^3]; steel
    E = 206e9 # Pa
    G = 81.5e9 # Pa

    Ei = np.array([E * A,  G * A, G * A])
    Fi = np.array([E * Ip, E * I3, E * I2])
    material_model = Hooke_quadratic(Ei, Fi)

    A_rho0 = rho * A
    B_rho0 = np.zeros(3) # symmetric cross section!
    C_rho0 = rho * np.diag(np.array([0, I3, I2]))

    ##########
    # gravity
    #########
    g = 9.81

    ###########################
    # discretization properties
    ###########################
    # p = 2
    p = 3
    # nQP = p + 1
    nQP = int(np.ceil((p**2 + 1) / 2)) + 1 # dynamics
    print(f'nQP: {nQP}')
    # nEl = 4 # 1 turn
    # nEl = 16 # 2 turns
    # nEl = 32 # 5 turns
    nEl = 64 # 10 turns
    # nEl = 128 # 20 turns

    #############################
    # fit reference configuration
    #############################
    coil_diameter = 32.0e-3 # 32mm
    coil_radius = coil_diameter / 2
    pitch_unloaded = 1.0e-3 # 1mm
    # turns = 0.5
    # turns = 2
    # turns = 5
    turns = 10
    # turns = 20
    nxi = 10000

    xi = np.linspace(0, turns, nxi)
    P, dP, ddP = helix3D(xi, coil_radius, pitch_unloaded)
    # Q = beam_reference_configuration(P, dP, ddP, p, nEl)
    # # Q = beam_reference_configuration(P, dP, ddP, p, nEl, debug=True)
    # compute directors using Serret-Frenet frame
    d1 = (dP.T / np.linalg.norm(dP, axis=-1)).T
    d2 = (ddP.T / np.linalg.norm(ddP, axis=-1)).T
    d3 = np.cross(d1, d2)

    # fit reference configuration
    qr0 = fit_B_Spline(P, p, nEl)
    qd1 = fit_B_Spline(d1, p, nEl)
    qd2 = fit_B_Spline(d2, p, nEl)
    qd3 = fit_B_Spline(d3, p, nEl)

    # ###############################
    # # debug reference configuration
    # ###############################
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # scale = 1.1 * coil_radius
    # ax.set_xlim3d(left=-scale, right=scale)
    # ax.set_ylim3d(bottom=-scale, top=scale)
    # ax.set_zlim3d(bottom=-scale, top=scale)

    # # ax.plot(*P.T, '-k')
    # length = 1.0e-3
    # # ax.quiver(*P.T, *d1.T, color='red', length=length, label='D1')
    # # ax.quiver(*P.T, *d2.T, color='green', length=length, label='D2')
    # # ax.quiver(*P.T, *d3.T, color='blue', length=length, label='D3')

    # ax.plot(*qr0.T, '--ob')
    # ax.quiver(*qr0.T, *qd1.T, color='red', length=length, label='D1_1')
    # ax.quiver(*qr0.T, *qd2.T, color='green', length=length, label='D2_1')
    # ax.quiver(*qr0.T, *qd3.T, color='blue', length=length, label='D3_1')

    # plt.show()
    # exit()

    Q = np.concatenate((qr0.T.reshape(-1), qd1.T.reshape(-1), qd2.T.reshape(-1), qd3.T.reshape(-1)))

    ########################
    # junction at the origin
    ########################
    r_OB1 = P[-1]
    A_IK1 = np.vstack((d1[-1], d2[-1], d3[-1])).T
    frame = Frame(r_OP=r_OB1, A_IK=A_IK1)

    ############
    # rigid body
    ############
    # TODO: which Theta is calculated here?
    m, Theta = Wilberforce_bob(coil_radius, debug=True)
    q0 = np.zeros(6)
    q0[2] = -49e-3 / 2 # center of mass is shifted!
    u0 = np.zeros(6)
    bob = Rigid_body_euler(m , Theta, q0=q0, u0=u0)
    
    ###################
    # solver parameters
    ###################
    n_load_steps = 10
    max_iter = 30
    tol = 1.0e-6

    t1 = 2.5e-2
    # dt = 1.0e-2 # beam as static force element implicit Euler
    dt = 5e-3 # full beam dynamics generalized alpha
    # dt = 5e-4 # full beam dynamics generalized alpha

    beam = Beam(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, q0=Q, Q=Q)

    model = Model()

    model.add(beam)
    model.add(frame)
    model.add(Rigid_connection(frame, beam, r_OB1, frame_ID2=(1,)))

    model.add(bob)
    r_OB = np.zeros(3)
    model.add(Rigid_connection(bob, beam, r_OB, frame_ID2=(0,)))

    if statics:
        model.add(Line_force(lambda xi, t: -t * g * A_rho0 * e3, beam))
        model.add(Force(lambda t: -t * g * m * e3, bob))
    else:
        model.add(Line_force(lambda xi, t: -g * A_rho0 * e3, beam))
        model.add(Force(lambda t: -g * m * e3, bob))

    model.assemble()

    if statics:
        solver = Newton(model, n_load_steps=n_load_steps, max_iter=max_iter, tol=tol)
    else:
        # build algebraic degrees of freedom indices for multiple beams
        tmp = int(beam.nu / 4)
        uDOF_algebraic = beam.uDOF[tmp:2*tmp] # include whole beam dynamics
        # uDOF_algebraic = beam.uDOF[tmp:4*tmp] # exclude centerline beam dynamics
        # uDOF_algebraic = beam.uDOF # beam as static force element
        # solver = Euler_backward_singular(model, t1, dt, uDOF_algebraic=uDOF_algebraic, numerical_jacobian=False, debug=False, newton_max_iter=20)

        # solver = Generalized_alpha_4_singular_index1(model, t1, dt, rho_inf=0.5, uDOF_algebraic=uDOF_algebraic, newton_tol=1.0e-6, numerical_jacobian=False)
        solver = Generalized_alpha_4_singular_index3(model, t1, dt, rho_inf=0.5, uDOF_algebraic=uDOF_algebraic, newton_tol=1.0e-6, numerical_jacobian=False)
        
    export_path = f'Wilberforce_pendulum_p{p}_nEL{nEl}_turns{turns}_t1{t1}_dt{dt}_c{c}'

    if save:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        sol = solver.solve()
        pr.disable()

        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats(0.1) # print only first 10% of the list

        sol.save(export_path)
    else:
        sol = load_solution(export_path)

    t = sol.t
    q = sol.q

    ########################
    # compute tip deflection
    ########################
    r0 = sol.q[0][beam.qDOF].reshape(12, -1)[:3, -1]
    dr = []
    for i, qi in enumerate(sol.q):
        dr.append( beam.centerline(qi)[:, -1] - r0)
        # dr.append( qi[beam.qDOF].reshape(12, -1)[:3, -1] - r0)
    dr = np.array(dr).T

    #####################################
    # TODO: export solution for each beam
    #####################################
    # fig = plt.figure()
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    ax3 = ax2.twinx()

    # visualize bob's displacements
    q_bob = q[:, bob.qDOF]
    ax2.plot(t, q_bob[:, 0], '-b', label='x')
    ax2.plot(t, q_bob[:, 1], '--b', label='y')
    ax2.plot(t, q_bob[:, 2], '-.b', label='z')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True)
    ax2.legend()

    # visualize rotation angles
    ax3.plot(t, q_bob[:, 3] * 180 / np.pi, '-r', label='phi_z')
    ax3.plot(t, q_bob[:, 4] * 180 / np.pi, '--r', label='phi_x')
    ax3.plot(t, q_bob[:, 5] * 180 / np.pi, '-.r', label='phi_y')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.grid(True)
    ax3.legend()
    
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    scale = 2 * coil_radius
    length_directors = 1.0e-2
    ax1.set_xlim3d(left=-scale, right=scale)
    ax1.set_ylim3d(bottom=-scale, top=scale)
    ax1.set_zlim3d(bottom=-2*scale, top=0)

    # prepare data for animation    
    slowmotion = 10
    fps = 10
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    center_line, = ax1.plot([], [], '-k')
    nodes, = ax1.plot([], [], '--ob')
    
    bob_com, = ax1.plot([], [], [], 'ok')
    d1_, = ax1.plot([], [], [], '-r')
    d2_, = ax1.plot([], [], [], '-g')
    d3_, = ax1.plot([], [], [], '-b')

    y_, = ax2.plot([], [], 'ob')
    phi_z_, = ax3.plot([], [], 'or')

    def animate(i):
        # animate beam centerline
        x, y, z = beam.centerline(q[i], n=400)
        center_line.set_data(x, y)
        center_line.set_3d_properties(z)

        # animate beam nodes
        x, y, z = q[i][beam.qDOF].reshape(12, -1)[:3]
        nodes.set_data(x, y)
        nodes.set_3d_properties(z)

        # animate rigid body
        x_S, y_S, z_S = bob.r_OP(t[i], q[i][bob.qDOF])    
        bob_com.set_data(np.array([x_S]), np.array([y_S]))
        bob_com.set_3d_properties(np.array([z_S]))

        d1, d2, d3 = bob.A_IK(t[i], q[i][bob.qDOF]).T * length_directors
        
        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        # animate y and phi_z
        y_.set_data(np.array([t[i]]), np.array([z_S]))
        phi_z_.set_data(np.array([t[i]]), np.array([q[i, bob.qDOF[3]] * 180 / np.pi]))

        return center_line, nodes, bob_com, d1_, d2_, d3_, y_, phi_z_

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()