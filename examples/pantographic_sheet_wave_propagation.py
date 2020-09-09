from cardillo.solver.solution import load_solution, save_solution
import numpy as np
from math import pi, sin, cos, exp, atan2, sqrt

from cardillo.model import Model
from cardillo.model.classical_beams.planar import Euler_bernoulli, Hooke, Inextensible_Euler_bernoulli
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw
from cardillo.solver.newton import Newton
from cardillo.solver import Generalized_alpha_index3_panto
from cardillo.discretization.B_spline import uniform_knot_vector
from cardillo.model.frame import Frame
from cardillo.math.algebra import A_IK_basic_z
from cardillo.utility.post_processing_vtk import post_processing


from cardillo.discretization.B_spline import B_spline_basis1D
class Junction():
    def __init__(self, beam1, beam2, la_g0=None):
        # rigid connection between to consecutive beams. End of beam1 is connected to start of beam2.
        self.nla_g = 3
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.beam1 = beam1
        self.beam2 = beam2

        self.frame_ID1 = (1,)
        self.frame_ID2 = (0,)
       
        N, N_xi = B_spline_basis1D(beam1.polynomial_degree, 1, beam1.knot_vector.data, 1).T
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)

        N, N_xi = B_spline_basis1D(beam2.polynomial_degree, 1, beam2.knot_vector.data, 0).T
        self.beam2_N = self.stack_shapefunctions(N, beam2.nq_el)
        self.beam2_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam2.nq_el)
        

    def assembler_callback(self):
        qDOF1 = self.beam1.qDOF_P(self.frame_ID1)
        qDOF2 = self.beam2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate([self.beam1.qDOF[qDOF1], self.beam2.qDOF[qDOF2]])
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2
        
        uDOF1 = self.beam1.uDOF_P(self.frame_ID1)
        uDOF2 = self.beam2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate([self.beam1.uDOF[uDOF1], self.beam2.uDOF[uDOF2]])
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

    def g(self, t, q):
        nq1 = self.nq1
        r_OP1 = self.beam1_N @ q[:nq1]
        r_OP2 = self.beam2_N @ q[nq1:]
        # tangent vector beam 1
        t = self.beam1_N_xi @ q[:nq1]
        # normal vector beam 2
        n = self.beam2_N_xi_perp @ q[nq1:]

        return np.concatenate([r_OP2 - r_OP1, [t @ n]]) 
        
    def g_q_dense(self, t, q):
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:2, :nq1] = - self.beam1_N
        g_q[:2, nq1:] = self.beam2_N

        # tangent vector beam 1
        t = self.beam1_N_xi @ q[:nq1]
        # normal vector beam 2
        n = self.beam2_N_xi_perp @ q[nq1:]

        g_q[2, :nq1] = n @ self.beam1_N_xi
        g_q[2, nq1:] = t @ self.beam2_N_xi_perp
        return g_q

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        # dense_num = Numerical_derivative(lambda t, q: self.g_q_dense(t, q).T @ la_g, order=2)._x(t, q)
        # [la_g[0], la_g[1]] @ (self.beam2_N - self.beam1_N) independent of q
        # [la_g[2] * self.beam1_N_xi.T @ n , la_g[2] * self.beam2_N_xi_perp.T @ t]
        nq1 = self.nq1
        nu1 = self.nu1

        dense = np.zeros((self._nu, self._nq))
        dense[:nu1, nq1:] = la_g[2] * self.beam1_N_xi.T @ self.beam2_N_xi_perp
        dense[nu1:, :nq1] = la_g[2] * self.beam2_N_xi_perp.T @ self.beam1_N_xi
        
        coo.extend( dense, (self.uDOF, self.qDOF))

    def stack_shapefunctions(self, N, nq_el):
        # return np.kron(np.eye(2), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

    def stack_shapefunctions_perp(self, N, nq_el):
        # return np.kron(np.array([[0, -1], [1, 0]]), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
        return NN

class Pivot_w_spring():
    def __init__(self, beam1, beam2, force_law, la_g0=None):
        # pivot between to consecutive beams. End of beam1 is connected to start of beam2.
        self.nla_g = 2
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.beam1 = beam1
        self.beam2 = beam2

        self.frame_ID1 = (1,)
        self.frame_ID2 = (0,)

        self.force_law = force_law
       
        N, N_xi = B_spline_basis1D(beam1.polynomial_degree, 1, beam1.knot_vector.data, 1).T
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)
        self.beam1_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam1.nq_el)

        N, N_xi = B_spline_basis1D(beam2.polynomial_degree, 1, beam2.knot_vector.data, 0).T
        self.beam2_N = self.stack_shapefunctions(N, beam2.nq_el)
        self.beam2_N_xi = self.stack_shapefunctions(N_xi, beam2.nq_el)
        self.beam2_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam2.nq_el)

    def assembler_callback(self):
        qDOF1 = self.beam1.qDOF_P(self.frame_ID1)
        qDOF2 = self.beam2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate([self.beam1.qDOF[qDOF1], self.beam2.qDOF[qDOF2]])
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2
        
        uDOF1 = self.beam1.uDOF_P(self.frame_ID1)
        uDOF2 = self.beam2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate([self.beam1.uDOF[uDOF1], self.beam2.uDOF[uDOF2]])
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

        q0_beam1 = self.beam1.q0[qDOF1]
        q0_beam2 = self.beam2.q0[qDOF2]

        t0_beam1 = self.beam1_N_xi @ q0_beam1
        theta0_beam1 = atan2(t0_beam1[1], t0_beam1[0])
        
        t0_beam2 = self.beam2_N_xi @ q0_beam2
        theta0_beam2 = atan2(t0_beam2[1], t0_beam2[0])

        # undeformed angle for torsional spring
        self.delta_theta0 = theta0_beam1 - theta0_beam2

        if self.force_law.g0 is None:
            self.force_law.g0 = self.delta_theta0

    def g(self, t, q):
        nq1 = self.nq1
        r_OP1 = self.beam1_N @ q[:nq1]
        r_OP2 = self.beam2_N @ q[nq1:]
        return r_OP2 - r_OP1
        
    def g_q_dense(self, t, q):
        return np.hstack([-self.beam1_N, self.beam2_N])

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        pass

    def stack_shapefunctions(self, N, nq_el):
        # return np.kron(np.eye(2), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

    def stack_shapefunctions_perp(self, N, nq_el):
        # return np.kron(np.array([[0, -1], [1, 0]]), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
        return NN

    def pot(self, t, q):
        return self.force_law.pot(t, self.__g(t, q))

    def f_pot(self, t, q):
        return - self.g_spring_q(t, q) * self.force_law.pot_g(t, self.g_spring(t, q))

    def g_spring(self, t, q):
        nq1 = self.nq1

        T1 = self.beam1_N_xi @ q[:nq1]
        T2 = self.beam2_N_xi @ q[nq1:]

        theta1 = atan2(T1[1], T1[0])
        theta2 = atan2(T2[1], T2[0])

        return theta1 - theta2

    def g_spring_q(self, t, q):
        nq1 = self.nq1
        T1 = self.beam1_N_xi @ q[:nq1]
        T2 = self.beam2_N_xi @ q[nq1:]
        
        g_q1 = (T1[0] * self.beam1_N_xi[1] - T1[1] * self.beam1_N_xi[0]) / (T1 @ T1)
        g_q2 = (T2[0] * self.beam2_N_xi[1] - T2[1] * self.beam2_N_xi[0]) / (T2 @ T2)

        W = np.hstack([g_q1, -g_q2])

        return W

    def f_pot_q(self, t, q, coo):
        # dense_num = Numerical_derivative(lambda t, q: self.f_pot(t, q), order=2)._x(t, q)
        dense = np.zeros((self._nu, self._nq))

        # current tangent vector
        nq1 = self.nq1
        T1 = self.beam1_N_xi @ q[:nq1]
        T2 = self.beam2_N_xi @ q[nq1:]

        # angle stiffness
        tmp1_1 = np.outer(self.beam1_N_xi[1], self.beam1_N_xi[0]) + np.outer(self.beam1_N_xi[0], self.beam1_N_xi[1])
        tmp1_2 = np.outer(self.beam1_N_xi[0], self.beam1_N_xi[0]) - np.outer(self.beam1_N_xi[1], self.beam1_N_xi[1])
        
        tmp2_1 = np.outer(self.beam2_N_xi[1], self.beam2_N_xi[0]) + np.outer(self.beam2_N_xi[0], self.beam2_N_xi[1])
        tmp2_2 = np.outer(self.beam2_N_xi[0], self.beam2_N_xi[0]) - np.outer(self.beam2_N_xi[1], self.beam2_N_xi[1])

        g_qq = np.zeros((self._nq, self._nq))
        g_qq[:nq1, :nq1] =   ((T1[1]**2 - T1[0]**2) * tmp1_1 + 2 * T1[0] * T1[1] * tmp1_2) / (T1 @ T1)**2
        g_qq[nq1:, nq1:] = - ((T2[1]**2 - T2[0]**2) * tmp2_1 + 2 * T2[0] * T2[1] * tmp2_2) / (T2 @ T2)**2
   
        W = self.g_spring_q(t, q)

        dense = - g_qq * self.force_law.pot_g(t, self.g_spring(t,q)) \
            - self.force_law.pot_gg(t, self.g_spring(t,q)) * np.outer(W, W)
                    
        coo.extend(dense, (self.uDOF, self.qDOF))

def create_pantograph(gamma, nRow, nCol, H, EA, EI, GI, A_rho0, p, nEl, nQP, r_OP_l, A_IK_l, r_OP_r, A_IK_r):
    
    assert p >= 2
    LBeam = H / (nRow * sin(gamma))
    L = nCol * LBeam * cos(gamma)

    ###################################################
    # create reference configuration individual beams #
    ###################################################
    
    # projections of beam length
    Lx = LBeam * cos(gamma)
    Ly = LBeam * sin(gamma)
    # upper left node
    xUL = 0         
    yUL = Ly*nRow
    # build reference configuration beam family 1
    nNd = nEl + p
    X0 = np.linspace(0, LBeam, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    Y1 = -np.copy(X0) * Ly / p
    X1 = X0 * Lx / p
    # build reference configuration beam family 2
    X2 = np.copy(X1)
    Y2 = -np.copy(Y1)
    
    #############
    # add beams #
    #############

    model = Model()

    material_model = Hooke(EA, EI)
    beams = []
    ID_mat = np.zeros((nRow, nCol)).astype(int)
    ID = 0
    for brow in range(0, nRow, 2):
        for bcol in range(0, nCol, 2):
            X = X1 + xUL + Lx * bcol
            Y = Y1 + yUL - Ly * brow

            # beam 1
            Q = np.concatenate([X, Y])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol] = ID
            ID += 1
            
            # beam 2
            Q = np.concatenate([X2 + X[-1], Y2 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol + 1] = ID
            ID += 1
      
    for brow in range(1, nRow, 2):
        for bcol in range(0, nCol, 2):
            X = X2 + xUL + Lx * bcol
            Y = Y2 + yUL - Ly * (brow + 1)
            # beam 1
            Q = np.concatenate([X, Y])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol] = ID
            ID += 1
            # beam 2
            Q = np.concatenate([X1 + X[-1], Y1 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol + 1] = ID
            ID += 1

    ######################################
    # add junctions within beam families #
    ######################################

    frame_ID1 = (1,)
    frame_ID2 = (0,)
            
    # odd colums
    for bcol in range(0, nCol, 2):
        for brow in range(0, nRow, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

    # even columns
    for bcol in range(1, nCol - 1, 2):
        for brow in range(1, nRow - 1, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

    ##########################################################
    # add pivots and torsional springs between beam families #
    ##########################################################
    
    # internal pivots
    for brow in range(0, nRow, 2):
        for bcol in range(0, nCol - 1):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            spring = Linear_spring(GI)
            model.add(Pivot_w_spring(beam1, beam2, spring))

    # lower boundary pivots
    for bcol in range(1, nCol - 1, 2):
        beam1 = beams[ID_mat[-1, bcol]]
        beam2 = beams[ID_mat[-1, bcol + 1]]
        r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
        spring = Linear_spring(GI)
        model.add(Pivot_w_spring(beam1, beam2, spring))

    ###########################
    # add boundary conditions #
    ###########################

    # clamping at the left hand side
    frame_l = Frame(r_OP=r_OP_l, A_IK=A_IK_l)
    model.add(frame_l)
    for idx in ID_mat[:, 0]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID2)], frame_ID=frame_ID2)
        model.add(Rigid_connection2D(frame_l, beam, r_OB, frame_ID2=frame_ID2))

    # clamping at the right hand side
    frame_r = Frame(r_OP=r_OP_r, A_IK = A_IK_r)
    model.add(frame_r)
    for idx in ID_mat[:, -1]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID1)], frame_ID=frame_ID1)
        model.add(Rigid_connection2D(beam, frame_r, r_OB, frame_ID1=frame_ID1))

    # assemble model
    model.assemble()

    return model, beams

if __name__ == "__main__":
    dynamics = True
    solveProblem = False
    t1 = 5e-2 / 150
    dt = 5e-2 / 1500
    rho_inf = 0.8

    # physical parameters
    gamma = pi/4
    nRow = 20
    nCol = 20

    H = 0.07
    LBeam = H / (nRow * sin(gamma))
    L = nCol * LBeam * cos(gamma)
    

    Yb = 500e6
    Gb = Yb / (2 * (1 + 0.4))
    a = 1.6e-3
    b = 1e-3
    rp = 0.45e-3
    hp = 1e-3

    Jg = (a * b**3) / 12
    
    EA = Yb * a * b
    EI = Yb * Jg
    GI = Gb * 0.5*(np.pi * rp**4)/hp

    A_rho0 = 930 * a * b

    displ = H / 5

    fcn = lambda t: displ * np.exp(-(t-0.004)**2/0.001**2)*(t*(t<0.001)+0.001*(t>=0.001))/0.001
    # fig, ax = plt.subplots()
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # x = linspace(0, t1, 1000)
    # y = []

    # for t in x:
    #     y.append(fcn(t))

    # ax.plot(x, y)
    # plt.show()

    rotationZ_l = 0 #-np.pi/10
    rotationZ_r = 0 #np.pi/10

    r_OP_l = lambda t: np.array([0, H / 2, 0]) + fcn(t) * np.array([1, 1, 0]) / sqrt(2)
    A_IK_l = lambda t: A_IK_basic_z(t * rotationZ_l)

    r_OP_r = lambda t: np.array([L, H / 2, 0])
    A_IK_r = lambda t: A_IK_basic_z(t * rotationZ_r)

    p = 2
    nQP = 4
    nEl = 1

    # create pantograph
    model, beams = create_pantograph(gamma, nRow, nCol, H, EA, EI, GI, A_rho0, p, nEl, nQP, r_OP_l, A_IK_l, r_OP_r, A_IK_r)

    # create .vtu file for initial configuration
    # post_processing(beams, np.array([0]), model.q0.reshape(1, model.q0.shape[0]), 'Pantograph_initial_configuration', binary=True)

    # choose solver
    if dynamics:
        solver = Generalized_alpha_index3_panto(model, t1, dt, rho_inf=rho_inf)
    else:
        solver = Newton(model, n_load_steps=5, max_iter=50, tol=1.0e-10, numerical_jacobian=False)
        

    if solveProblem == True:
        # import cProfile, pstats
        # pr = cProfile.Profile()
        # pr.enable()
        sol = solver.solve()
        # pr.disable()

        # sortby = 'cumulative'
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.print_stats(0.1) # print only first 10% of the list
        if dynamics == True:
            save_solution(sol, f'PantographicSheet_{nRow}x{nCol}_dynamics')
        else:
            save_solution(sol, f'PantographicSheet_{nRow}x{nCol}_statics')
    else:
        if dynamics == True:
            sol = load_solution(f'PantographicSheet_{nRow}x{nCol}_dynamics')
        else:
            sol = load_solution(f'PantographicSheet_{nRow}x{nCol}_statics')

    if dynamics:
        post_processing(beams, sol.t[::5], sol.q[::5], f'PantographicSheet_{nRow}x{nCol}_dynamics', u = sol.u[::5], binary=True)
    else:
        post_processing(beams, sol.t, sol.q, f'PantographicSheet_{nRow}x{nCol}_statics', binary=True)