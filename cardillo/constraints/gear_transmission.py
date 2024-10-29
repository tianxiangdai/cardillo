import numpy as np
from .revolute import Revolute


class GearTransmission:
    def __init__(
        self,
        subsystem1: Revolute,
        subsystem2: Revolute,
        radius1,
        radius2,
        reversed=False,
    ):
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.radius1 = radius1
        self.radius2 = radius2 if not reversed else -radius2
        self.nla_g = 1

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        self._nq1 = self.subsystem1._nq
        self._nq2 = self.subsystem2._nq
        self._nq = self._nq1 + self._nq2
        self.qDOF = np.concatenate((qDOF1, qDOF2))

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        self._nu1 = self.subsystem1._nu
        self._nu2 = self.subsystem2._nu
        self._nu = self._nu1 + self._nu2
        self.uDOF = np.concatenate((uDOF1, uDOF2))

    def g(self, t, q):
        nq1 = self._nq1
        return (
            self.subsystem1.l(t, q[:nq1]) - self.subsystem1.angle0
        ) * self.radius1 - (
            self.subsystem2.l(t, q[nq1:]) - self.subsystem2.angle0
        ) * self.radius2

    def g_q(self, t, q):
        nq1 = self._nq1
        g_q = np.zeros(self._nq, dtype=q.dtype)
        g_q[:nq1] = self.subsystem1.l_q(t, q[:nq1]) * self.radius1
        g_q[nq1:] = -self.subsystem2.l_q(t, q[nq1:]) * self.radius2
        return g_q

    def g_dot(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        return (
            self.subsystem1.l_dot(t, q[:nq1], u[:nu1]) * self.radius1
            - self.subsystem2.l_dot(t, q[nq1:], u[nu1:]) * self.radius2
        )

    def g_dot_q(self, t, q, u):
        nq1 = self._nq1
        nu1 = self._nu1
        g_dot_q = np.zeros(self._nq, dtype=q.dtype)
        g_dot_q[:nq1] = self.subsystem1.l_dot_q(t, q[:nq1], u[:nu1]) * self.radius1
        g_dot_q[nq1:] = -self.subsystem2.l_dot_q(t, q[nq1:], u[nu1:]) * self.radius2
        return g_dot_q

    def g_dot_u(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        g_dot_u = np.zeros(self._nu, dtype=q.dtype)
        g_dot_u[:nu1] = self.subsystem1.l_dot_u(t, q[:nq1], None) * self.radius1
        g_dot_u[nu1:] = -self.subsystem2.l_dot_u(t, q[nq1:], None) * self.radius2
        return g_dot_u

    def g_ddot(self, t, q, u, u_dot):
        nq1 = self._nq1
        nu1 = self._nu1
        e_c1_1 = self.subsystem1.A_IJ1(t, q[:nq1])[:, self.subsystem1.axis]
        e_c1_2 = self.subsystem2.A_IJ1(t, q[nq1:])[:, self.subsystem2.axis]
        g_ddot = (
            self.subsystem1.Psi2(t, q[:nq1], u[:nu1], u_dot[:nu1])
            - self.subsystem1.Psi1(t, q[:nq1], u[:nu1], u_dot[:nu1])
        ) @ e_c1_1 * self.radius1 - (
            self.subsystem2.Psi2(t, q[nq1:], u[nu1:], u_dot[nu1:])
            - self.subsystem2.Psi1(t, q[nq1:], u[nu1:], u_dot[nu1:])
        ) @ e_c1_2 * self.radius2
        return g_ddot

    # def g_ddot_q(self, t, q, u, u_dot):
    #     return

    # def g_ddot_u(self, t, q, u, u_dot):
    #     return

    def W_g(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        W_g = np.zeros((self._nu, 1), dtype=q.dtype)
        W_g[:nu1] = self.subsystem1.W_l(t, q[:nq1]) * self.radius1
        W_g[nu1:] = -self.subsystem2.W_l(t, q[nq1:]) * self.radius2
        return W_g

    def Wla_g_q(self, t, q, la_g):
        nq1 = self._nq1
        nu1 = self._nu1
        nq2 = self._nq2
        nu2 = self._nu2
        Wla_g_q = np.zeros((self._nu, self._nq), dtype=q.dtype)
        Wla_g_q[:nu1, :nq1] = (
            self.subsystem1.W_l_q(t, q[:nq1]).reshape((nu1, nq1)) * self.radius1 * la_g
        )
        Wla_g_q[nu1:, nq1:] = (
            -self.subsystem2.W_l_q(t, q[nq1:]).reshape((nu2, nq2)) * self.radius2 * la_g
        )
        return Wla_g_q
