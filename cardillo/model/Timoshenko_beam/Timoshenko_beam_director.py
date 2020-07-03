import numpy as np

# from cardillo.discretization.quadrature.gauss import gauss
# from cardillo.discretization.polynomials.bspline import uniform_knotvector, B_spline_basis, find_Knotspan, draw_B_spline3D
# from cardillo.utility.algebra import norm3, skew2ax, cross3 #, LeviCivita3, cross_nm
from cardillo.math.numerical_derivative import Numerical_derivative

class Timoshenko_beam_director():
    def __init__(self, material_model, A_rho0, B_rho0, C_rho0, \
                 p, nQP, nEl, Q, q0=None, u0=None):

        # beam properties
        self.materialModel = material_model # material model
        self.A_rho = A_rho0 # line density
        self.B_rho = B_rho0 # first moment
        self.C_rho = C_rho0 # second moment

        # discretization parameters
        self.p = p             # polynomial degree
        self.nQP = nQP         # number of quadrature points
        self.nEl = nEl         # number of elements

        self.qp, self.qw = gauss(nQP) # quadrature points and quadrature weights for Gauss-Legendre quadrature rule
        self.kv = kv = uniform_knotvector(p, nEl) # uniform open knot vector
        self.nNd = nNd = nEl + p # number of nodes, depending on the used shape functions
        self.nNdEl = nNdEl = p + 1 # number of nodes given in a single element
        self.nNDOF = nNDOF = 3 + 3 * 3 # number of degrees of freedom at a single node: 3 for the position vector and 3 * 3 for the three directors

        self.nq = self.nNDOF * self.nNd # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = self.nNDOF * nNdEl # total number of generalized coordinates in a single element

        # compute allocation matrices
        tmp1 = np.tile(np.arange(0, nNdEl), nNDOF)
        tmp2 = np.repeat(np.arange(0, nNDOF * nNd, step=nNd), nNdEl)
        self.elDOF = (np.zeros((nNDOF * nNdEl, nEl), dtype=int) + np.arange(nEl)).T + tmp1 + tmp2

        tmp3 = (np.zeros((self.nNDOF, nNd), dtype=int) + np.arange(nNd)).T
        tmp4 = np.tile(np.arange(0, nNDOF * nNd, step=nNd), nNd).reshape((nNd, nNDOF))
        self.nodalDOF = tmp3 + tmp4

        # compute B-spline shape functions
        self.shapeFunctions = np.empty((nEl, 2, nQP, nNdEl))
        for el in range(nEl):
            deltaXi = kv[el + p + 1] - kv[el + p]
            xi = kv[el + p] + deltaXi * (self.qp + 1) / 2
            self.shapeFunctions[el] = B_spline_basis(kv, p, 1, xi)
        self.shapeFunctionsBdry = B_spline_basis(kv, p, 1, np.array([0, 1]))
            
        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        # self.la_g0 = np.zeros(self.nla_g) if la0 is None else la0

        # degrees of freedom on the element level
        self.DOF_r = np.arange(3 * nNdEl)
        self.DOF_d1 = np.arange(3 * nNdEl, 6  * nNdEl)
        self.DOF_d2 = np.arange(6 * nNdEl, 9  * nNdEl)
        self.DOF_d3 = np.arange(9 * nNdEl, 12 * nNdEl)

    # def __init__(self, V_rho, B_rho0, C_rho0, q0=None, u0=None, la0=None):
    #     self.nq = 12
    #     self.nu = 12
    #     self.nla_g = 6
    #     self.V_rho = V_rho
    #     self.B_rho0 = B_rho0
    #     self.C_rho0 = C_rho0

    #     if q0 is None:
    #         e1 = np.array([1, 0, 0])
    #         e2 = np.array([0, 1, 0])
    #         e3 = np.array([0, 0, 1])
    #         self.q0 = np.hstack((np.zeros(3), e1, e2, e3))
    #     else:
    #         self.q0 = q0

    #     self.u0 = np.zeros(self.nu) if u0 is None else u0
    #     self.la_g0 = np.zeros(self.nla_g) if la0 is None else la0

    #     self.M_ = np.zeros((self.nu, self.nu))
    #     self.M_[:3, :3] = np.eye(3) * self.V_rho
    #     self.M_[:3, 3:6] = np.eye(3) * self.B_rho0[0]
    #     self.M_[:3, 6:9] = np.eye(3) * self.B_rho0[1]
    #     self.M_[:3, 9:12] = np.eye(3) * self.B_rho0[2]
        
    #     self.M_[3:6, :3] = np.eye(3) * self.B_rho0[0]
    #     self.M_[3:6, 3:6] = np.eye(3) * self.C_rho0[0, 0]
    #     self.M_[3:6, 6:9] = np.eye(3) * self.C_rho0[0, 1]
    #     self.M_[3:6, 9:12] = np.eye(3) * self.C_rho0[0, 2]
        
    #     self.M_[6:9, :3] = np.eye(3) * self.B_rho0[1]
    #     self.M_[6:9, 3:6] = np.eye(3) * self.C_rho0[1, 0]
    #     self.M_[6:9, 6:9] = np.eye(3) * self.C_rho0[1, 1]
    #     self.M_[6:9, 9:12] = np.eye(3) * self.C_rho0[1, 2]

    #     self.M_[9:12, :3] = np.eye(3) * self.B_rho0[2]
    #     self.M_[9:12, 3:6] = np.eye(3) * self.C_rho0[2, 0]
    #     self.M_[9:12, 6:9] = np.eye(3) * self.C_rho0[2, 1]
    #     self.M_[9:12, 9:12] = np.eye(3) * self.C_rho0[2, 2]

    # #########################################
    # # equations of motion
    # #########################################
    # def M(self, t, q, M_coo):
    #     M_coo.extend(self.M_, (self.uDOF, self.uDOF))

    # #########################################
    # # kinematic equation
    # #########################################
    # def q_dot(self, t, q, u):
    #     return u

    # def q_dot_q(self, t, q, u, coo):
    #     coo.extend(np.zeros((self.nq, self.nq)), (self.qDOF, self.qDOF))

    # def B(self, t, q, coo):
    #     coo.extend(np.eye(self.nq, self.nu), (self.qDOF, self.uDOF))

    # #########################################
    # # bilateral constraints on position level
    # #########################################
    # def g(self, t, q):
    #     d1 = q[3:6]
    #     d2 = q[6:9]
    #     d3 = q[9:]

    #     gap = np.zeros(self.nla_g)
    #     gap[0] = d1 @ d1 - 1
    #     gap[1] = d2 @ d2 - 1
    #     gap[2] = d3 @ d3 - 1
    #     gap[3] = d1 @ d2
    #     gap[4] = d1 @ d3
    #     gap[5] = d2 @ d3

    #     return gap

    # def g_q_dense(self, t, q):

    #     d1 = q[3:6]
    #     d2 = q[6:9]
    #     d3 = q[9:]

    #     gap_q = np.zeros((self.nla_g, self.nq))
    #     gap_q[0, 3:6] = 2 * d1
    #     gap_q[1, 6:9] = 2 * d2
    #     gap_q[2, 9:12] = 2 * d3

    #     gap_q[3, 3:6] = d2
    #     gap_q[3, 6:9] = d1

    #     gap_q[4, 3:6] = d3
    #     gap_q[4, 9:12] = d1

    #     gap_q[5, 6:9] = d3
    #     gap_q[5, 9:12] = d2

    #     # gap_q_num = NumericalDerivativeNew(self.gap_dense, order=2).dR_dq(t, q)
    #     # diff = gap_q - gap_q_num
    #     # np.set_printoptions(precision=3)
    #     # error = np.linalg.norm(diff)
    #     # print(f'error num_tan - tan = {error}')
    #     # return gap_q_num

    #     return gap_q
    
    # def g_qq_dense(self, t, q):

    #     gap_qq = np.zeros((self.nla_g, self.nq, self.nq))
    #     gap_qq[0, 3:6, 3:6] = 2 * np.eye(3)
    #     gap_qq[1, 6:9, 6:9] = 2 * np.eye(3)
    #     gap_qq[2, 9:12, 9:12] = 2 * np.eye(3)
        
    #     gap_qq[3, 3:6, 6:9] = np.eye(3)
    #     gap_qq[3, 6:9, 3:6] = np.eye(3)
        
    #     gap_qq[4, 3:6, 9:12] = np.eye(3)
    #     gap_qq[4, 9:12, 3:6] = np.eye(3)
        
    #     gap_qq[5, 6:9, 9:12] = np.eye(3)
    #     gap_qq[5, 9:12, 6:9] = np.eye(3)

    #     # gap_qq_num = NumericalDerivativeNew(self.gap_q_dense, order=2).dR_dq(t, q)
    #     # diff = gap_qq - gap_qq_num
    #     # np.set_printoptions(precision=3)
    #     # error = np.linalg.norm(diff)
    #     # print(f'error num_tan - tan = {error}')
    #     # return gap_qq_num

    #     return gap_qq

    # def g_q(self, t, q, coo):
    #     coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))
   
    # def W_g(self, t, q, coo):
    #     coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    # def Wla_g_q(self, t, q, la_g, coo):
    #     dense = np.einsum('ijk,i->jk', self.g_qq_dense(t, q), la_g)
    #     coo.extend(dense, (self.uDOF, self.qDOF))

    # #########################################
    # # helper functions
    # #########################################
    # def qDOF_P(self, frame_ID=None):
    #     return self.qDOF

    # def uDOF_P(self, frame_ID=None):
    #     return self.uDOF

    # def A_IK(self, t, q, frame_ID=None):
    #     return np.vstack((q[3:6], q[6:9], q[9:12])).T

    # def A_IK_q(self, t, q, frame_ID=None):
    #     A_IK_q = np.zeros((3, 3, self.nq))
    #     A_IK_q[:, 0, 3:6] = np.eye(3)
    #     A_IK_q[:, 1, 6:9] = np.eye(3)
    #     A_IK_q[:, 2, 9:12] = np.eye(3)
    #     return A_IK_q

    # def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
    #     return q[:3] + self.A_IK(t, q) @ K_r_SP

    # def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
    #     r_OP_q = np.zeros((3, self.nq))
    #     r_OP_q[:, :3] = np.eye(3)
    #     r_OP_q[:, :] += np.einsum('ijk,j->ik', self.A_IK_q(t, q), K_r_SP)
    #     return r_OP_q

    #     # r_OP_q_num = Numerical_derivative(lambda t, q: self.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP))._x(t, q)
    #     # diff = r_OP_q_num - r_OP_q
    #     # error = np.max(np.abs(diff))
    #     # print(f'error in r_OP_q = {error}') 
    #     # return r_OP_q_num

    # def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
    #     return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP) @ u

    # def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
    #     return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)

    # def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
    #     return np.zeros((3, self.nu, self.nq))

    # # def K_J_R(self, t, q, frame_ID=None):
    # #     J_R = np.zeros((3, self.nu))
    # #     J_R[:, 3:] = np.eye(3)
    # #     return J_R

    # # def K_J_R_q(self, t, q, frame_ID=None):
    # #     return np.zeros((3, self.nu, self.nq))

if __name__ == "__main__":
    pass