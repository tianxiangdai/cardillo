import numpy as np
from scipy.integrate import solve_ivp

from cardillo.discrete import Box, Sphere, RigidBody
from cardillo.system import System
from cardillo.forces import Force
from cardillo.constraints import FixedDistance, Prismatic
from cardillo.solver import ScipyIVP

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy

from my_interfaces.msg import CartPoleState, Forcing

test_cardillo = True

m_cart = 1
m_pole = 1
g_accel = 9.81
l2 = 0.1
alpha0 = np.pi / 4
fps = 1000


class CartPole:
    def __init__(self):
        self.fps = fps

        self.system = System()
        self.cart = Box(RigidBody)(
            np.array([30, 10, 10]) * 1e-3, mass=m_cart, B_Theta_C=np.eye(3)
        )
        # self.force = Force(lambda t: np.array([0, 0, 0], dtype=np.float64), self.cart)
        self.pole = Sphere(RigidBody)(
            radius=10e-3,
            mass=m_pole,
            B_Theta_C=np.eye(3),
            q0=np.array([l2 * np.sin(alpha0), -l2 * np.cos(alpha0), 0, 1, 0, 0, 0]),
        )
        rc1 = Prismatic(self.cart, self.system.origin, axis=0)
        rc2 = FixedDistance(self.cart, self.pole)
        grav = Force(np.array([0, -m_pole * g_accel, 0]), self.pole)
        self.forcing = Force(np.zeros(3, dtype=np.float64), self.cart)

        for el in [self.cart, self.pole, self.forcing, grav, rc1, rc2]:
            self.system.add(el)
        self.system.assemble()
        self.solver = ScipyIVP(self.system, 1 / self.fps, 1 / self.fps)
        assert len(self.solver.t_eval) == 2

    def set_force(self, la):
        self.forcing.force = lambda t, la=la: np.array([la, 0, 0], dtype=np.float64)

    def set_solver(self, q0, u0):
        self.system.set_new_initial_state(q0, u0)
        self.solver.x0 = np.concatenate([q0, u0])

    def solve(self):
        return self.solver.solve()


# scipy ivp
M_inv = lambda alpha: np.array(
    [
        [m_pole * l2**2, -m_pole * l2 * np.cos(alpha)],
        [-m_pole * l2 * np.cos(alpha), m_cart + m_pole],
    ]
    / (m_cart + m_pole * np.sin(alpha) ** 2)
    / (m_pole * l2**2)
)
h = lambda alpha, dalpha: m_pole * l2 * np.sin(alpha) * np.array([dalpha**2, -g_accel])


class SimulatorNode(Node):

    def __init__(self):
        super().__init__("simulator")
        # subscription
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Forcing, "forcing", self.callback_forcing, qos_profile)
        # model
        if test_cardillo:
            self.cart_pole = CartPole()
        # publisher
        self.publisher = self.create_publisher(CartPoleState, "cart_pole_state", 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.__la = 0.0
        self.__ivp_y0 = np.array([0, alpha0, 0, 0])

    def callback_forcing(self, msg_forcing):
        self.__la = msg_forcing.la

    def timer_callback(self):
        # step simulation
        if test_cardillo:
            cart = self.cart_pole.cart
            pole = self.cart_pole.pole
            self.cart_pole.set_force(self.__la)
            sol = self.cart_pole.solve()
            ti, qi, ui = sol.t[-1], sol.q[-1], sol.u[-1]
            self.cart_pole.set_solver(qi, ui)
            # update state
            message = CartPoleState()
            message.la = self.__la
            message.r_os_cart = cart.r_OP(ti, qi[cart.qDOF])
            message.r_os_pole = pole.r_OP(ti, qi[pole.qDOF])
            message.v_s_cart = cart.v_P(ti, qi[cart.qDOF], ui[cart.uDOF])
            message.v_s_pole = pole.v_P(ti, qi[pole.qDOF], ui[pole.uDOF])
            self.publisher.publish(message)
        else:

            def fun(t, y):
                x, alpha, dx, dalpha = y
                dy = np.empty_like(y)
                dy[:2] = y[2:]
                dy[2:] = M_inv(alpha) @ (h(alpha, dalpha) + np.array([self.__la, 0]))
                return dy

            s = solve_ivp(
                fun,
                method="Radau",
                t_span=[0, 1 / fps],
                y0=self.__ivp_y0,
                t_eval=[0, 1 / fps],
            )
            self.__ivp_y0 = s.y[:, -1]
            x_ivp, alpha_ivp, dx_ivp, dalpha_ivp = s.y[:, -1]
            # update state
            message = CartPoleState()
            message.la = self.__la
            message.r_os_cart = np.array([x_ivp, 0.0, 0.0])
            message.r_os_pole = [
                x_ivp + l2 * np.sin(alpha_ivp),
                -l2 * np.cos(alpha_ivp),
                0.0,
            ]
            message.v_s_cart = [dx_ivp, 0.0, 0.0]
            message.v_s_pole = [
                dx_ivp + l2 * np.cos(alpha_ivp) * dalpha_ivp,
                l2 * np.sin(alpha_ivp) * dalpha_ivp,
                0.0,
            ]
            self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        node.destroy_node()
        # rclpy.shutdown()


if __name__ == "__main__":
    main()
