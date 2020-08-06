import numpy as np
from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.math.algebra import cross3, ax2skew

class Sphere_to_plane():
    def __init__(self, frame, subsystem, r, mu, prox_r_N, prox_r_T, e_N=None, e_T=None, frame_ID=np.zeros(3), K_r_SP=np.zeros(3), la_N0=None, la_T0=None):
        
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.mu = np.array([mu])
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_T = np.array([prox_r_T])

        self.nla_N = 1

        if mu == 0:
            self.nla_T = 0
            self.NT_connectivity = [[]]
        else:
            self.nla_T =  2 * self.nla_N 
            self.NT_connectivity = [ [0, 1] ]
            self.gamma_T = self.__gamma_T
            
        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_T = np.zeros(self.nla_N) if e_T is None else np.array([e_T])
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.t1t2 = lambda t: self.frame.A_IK(t).T[:2]
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        self.v_Q = lambda t: self.frame.v_P(t)
        self.a_Q = lambda t: self.frame.a_P(t)

        self.K_r_SP = K_r_SP 

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_T0 = np.zeros(self.nla_T) if la_T0 is None else la_T0

        self.is_assembled = False

    def assembler_callback(self):
        qDOF = self.subsystem.qDOF_P(self.frame_ID)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.uDOF_P(self.frame_ID)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        self.r_OP = lambda t, q: self.subsystem.r_OP(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.r_OP_q = lambda t, q: self.subsystem.r_OP_q(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.v_P = lambda t, q, u: self.subsystem.v_P(t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.J_P = lambda t, q: self.subsystem.J_P(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.J_P_q = lambda t, q: self.subsystem.J_P_q(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        
        self.Omega = lambda t, q, u: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        self.J_R = lambda t, q: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        self.J_R_q = lambda t, q: np.einsum('ijl,jk->ikl', self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID), self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)) + np.einsum('ij,jkl->ikl', self.subsystem.A_IK(t, q, frame_ID=self.frame_ID), self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID))
        self.Psi = lambda t, q, u, a: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)

        self.is_assembled = True

    def g_N(self, t, q):
        return np.array([self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t))]) - self.r

    def g_N_q_dense(self, t, q):
        return np.array([self.n(t) @ self.r_OP_q(t, q)])

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))])

    def g_N_dot_q_dense(self, t, q, u):
        return np.array([self.n(t) @ self.v_P_q(t, q, u) ])

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ self.J_P(t, q)])
    
    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_N_q(self, t, q, u_pre, u_post, coo):
        g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
        g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
        dense = g_N_q_post + self.e_N * g_N_q_pre
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array([self.n(t) @ (self.a_P(t, q, u, u_dot) - self.a_Q(t))])

    def g_N_ddot_q(self, t, q, u, u_dot, coo):
        dense = np.array([self.n(t) @ self.a_P_q(t, q, u, u_dot)])
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def g_N_ddot_u(self, t, q, u, u_dot, coo):
        dense = np.array([self.n(t) @ self.a_P_u(t, q, u, u_dot)])
        coo.extend(dense, (self.la_NDOF, self.uDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        dense = la_N[0] * np.einsum('i,ijk->jk', self.n(t), self.J_P_q(t, q))
        # dense_num = np.einsum('i,ijk->jk', la_N, Numerical_derivative(self.g_N_dot_u_dense, order=2)._x(t, q))
        # error = np.linalg.norm(dense - dense_num)
        # print(f'error: {error}')
        coo.extend(dense, (self.uDOF, self.qDOF))

    def __gamma_T(self, t, q, u):
        v_C = self.v_P(t, q, u) + self.r * cross3(self.n(t), self.Omega(t, q, u))
        return self.t1t2(t) @ (v_C - self.v_Q(t))

    def gamma_T_dot(self, t, q, u, u_dot):
        # #TODO: t1t2_dot(t) & n_dot(t)
        Omega = self.Omega(t, q, u)
        r_PC = -self.r * self.n(t)
        a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC) + cross3(Omega, cross3(Omega, r_PC))
        gamma_T_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        return gamma_T_dot

        # gamma_T_q = Numerical_derivative(self.gamma_T, order=2)._x(t, q, u)
        # gamma_T_u = self.gamma_T_u_dense(t, q)
        # gamma_T_dot_num = gamma_T_q @ self.subsystem.q_dot(t, q, u) + gamma_T_u @ u_dot
        # error = np.linalg.norm(gamma_T_dot_num - gamma_T_dot)
        # print(f'error: {error}')
        # return gamma_T_dot_num

    def gamma_T_u_dense(self, t, q):
        J_C = self.J_P(t, q) + self.r * ax2skew(self.n(t)) @ self.J_R(t, q)
        return self.t1t2(t) @ J_C

    def W_T(self, t, q, coo):
        coo.extend(self.gamma_T_u_dense(t, q).T, (self.uDOF, self.la_TDOF))

    def Wla_T_q(self, t, q, la_T, coo):
        J_C_q = self.J_P_q(t, q) + self.r * np.einsum('ij,jkl->ikl', ax2skew(self.n(t)), self.J_R_q(t, q))
        dense = np.einsum('i,ij,jkl->kl', la_T, self.t1t2(t), J_C_q)
        coo.extend(dense, (self.uDOF, self.qDOF))
        # dense_num = np.einsum('i,ijk->jk', la_T, Numerical_derivative(self.gamma_T_u_dense, order=2)._x(t, q))
        # error = np.linalg.norm(dense - dense_num)
        # print(f'error: {error}')
        # coo.extend(dense_num, (self.uDOF, self.qDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_T(self, t, q, u_pre, u_post):
        return self.gamma_T(t, q, u_post) + self.e_T * self.gamma_T(t, q, u_pre)
        
class Sphere_to_plane2D():
    def __init__(self, frame, subsystem, r, mu, prox_r_N, prox_r_T, e_N=None, e_T=None, frame_ID=np.zeros(3), K_r_SP=np.zeros(3), la_N0=None, la_T0=None):
        
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.mu = np.array([mu])
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_T = np.array([prox_r_T])

        self.nla_N = 1

        if mu == 0:
            self.nla_T = 0
            self.NT_connectivity = [[]]
        else:
            self.nla_T = self.nla_N 
            self.NT_connectivity = [ [0] ]
            self.gamma_T = self.__gamma_T
            
        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_T = np.zeros(self.nla_N) if e_T is None else np.array([e_T])
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.t = lambda t: self.frame.A_IK(t).T[0]
        self.n = lambda t: self.frame.A_IK(t)[:, 1]
        self.v_Q = lambda t: self.frame.v_P(t)
        self.a_Q = lambda t: self.frame.a_P(t)

        self.K_r_SP = K_r_SP 

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_T0 = np.zeros(self.nla_T) if la_T0 is None else la_T0

        self.is_assembled = False

    def assembler_callback(self):
        qDOF = self.subsystem.qDOF_P(self.frame_ID)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.uDOF_P(self.frame_ID)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        self.r_OP = lambda t, q: self.subsystem.r_OP(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.r_OP_q = lambda t, q: self.subsystem.r_OP_q(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.v_P = lambda t, q, u: self.subsystem.v_P(t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.J_P = lambda t, q: self.subsystem.J_P(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.J_P_q = lambda t, q: self.subsystem.J_P_q(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
        
        self.Omega = lambda t, q, u: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        self.J_R = lambda t, q: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        self.J_R_q = lambda t, q: np.einsum('ijl,jk->ikl', self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID), self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)) + np.einsum('ij,jkl->ikl', self.subsystem.A_IK(t, q, frame_ID=self.frame_ID), self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID))
        self.Psi = lambda t, q, u, a: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)

        self.is_assembled = True

    def g_N(self, t, q):
        return np.array([self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t))]) - self.r

    def g_N_q_dense(self, t, q):
        return np.array([self.n(t) @ self.r_OP_q(t, q)])

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))])

    def g_N_dot_q_dense(self, t, q, u):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ self.v_P_q(t, q, u) ])

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ self.J_P(t, q)])
    
    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_N_q(self, t, q, u_pre, u_post, coo):
        g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
        g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
        dense = g_N_q_post + self.e_N * g_N_q_pre
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array([self.n(t) @ (self.a_P(t, q, u, u_dot) - self.a_Q(t))])

    def g_N_ddot_q(self, t, q, u, u_dot, coo):
        dense = np.array([self.n(t) @ self.a_P_q(t, q, u, u_dot)])
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def g_N_ddot_u(self, t, q, u, u_dot, coo):
        dense = np.array([self.n(t) @ self.a_P_u(t, q, u, u_dot)])
        coo.extend(dense, (self.la_NDOF, self.uDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        dense = la_N[0] * np.einsum('i,ijk->jk', self.n(t), self.J_P_q(t, q))
        # dense_num = np.einsum('i,ijk->jk', la_N, Numerical_derivative(self.g_N_dot_u_dense, order=2)._x(t, q))
        # error = np.linalg.norm(dense - dense_num)
        # print(f'error: {error}')
        coo.extend(dense, (self.uDOF, self.qDOF))

    def __gamma_T(self, t, q, u):
        v_C = self.v_P(t, q, u) + self.r * cross3(self.n(t), self.Omega(t, q, u))
        return np.array([self.t(t) @ (v_C - self.v_Q(t))])

    # TODO
    def gamma_T_dot(self, t, q, u, u_dot):
        Omega = self.Omega(t, q, u)
        r_PC = -self.r * self.n(t)
        a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC) + cross3(Omega, cross3(Omega, r_PC))
        gamma_T_dot = self.t(t) @ (a_C - self.a_Q(t))
        return gamma_T_dot

        # gamma_T_q = Numerical_derivative(self.gamma_T, order=2)._x(t, q, u)
        # gamma_T_u = self.gamma_T_u_dense(t, q)
        # gamma_T_dot_num = gamma_T_q @ self.subsystem.q_dot(t, q, u) + gamma_T_u @ u_dot
        # error = np.linalg.norm(gamma_T_dot_num - gamma_T_dot)
        # print(f'error: {error}')
        # return gamma_T_dot_num

    def gamma_T_u_dense(self, t, q):
        J_C = self.J_P(t, q) + self.r * ax2skew(self.n(t)) @ self.J_R(t, q)
        return np.array([self.t(t) @ J_C])

    def W_T(self, t, q, coo):
        coo.extend(self.gamma_T_u_dense(t, q).T, (self.uDOF, self.la_TDOF))

    def Wla_T_q(self, t, q, la_T, coo):
        J_C_q = self.J_P_q(t, q) + self.r * np.einsum('ij,jkl->ikl', ax2skew(self.n(t)), self.J_R_q(t, q))
        dense = la_T[0] * np.einsum('i,ikl->kl', self.t(t), J_C_q)
        coo.extend(dense, (self.uDOF, self.qDOF))

        # dense_num = np.einsum('i,ijk->jk', la_T, Numerical_derivative(self.gamma_T_u_dense, order=2)._x(t, q))
        # error = np.linalg.norm(dense - dense_num)
        # print(f'error: {error}')
        # coo.extend(dense_num, (self.uDOF, self.qDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_T(self, t, q, u_pre, u_post):
        return self.gamma_T(t, q, u_post) + self.e_T * self.gamma_T(t, q, u_pre)