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

from my_interfaces.msg import CartBallState, Forcing


m_cart = 1
m_ball = 1
g_accel = 9.81
l2 = 0.1
alpha0 = np.pi / 4


class CartBall:
    def __init__(self):
        self.fps = 1000
        system = System()
        self.cart = Box(RigidBody)(
            np.array([30, 10, 10]) * 1e-3, mass=m_cart, B_Theta_C=np.eye(3)
        )
        self.forcing = Force(lambda t: np.array([0, 0, 0], dtype=np.float64), self.cart)
        self.ball = Sphere(RigidBody)(
            radius=10e-3,
            mass=m_ball,
            B_Theta_C=np.eye(3),
            q0=np.array([l2 * np.sin(alpha0), -l2 * np.cos(alpha0), 0, 1, 0, 0, 0]),
        )
        rc1 = Prismatic(self.cart, system.origin, axis=0)
        rc2 = FixedDistance(self.cart, self.ball)
        grav = Force(np.array([0, -m_ball * g_accel, 0]), self.ball)

        for el in [self.cart, self.forcing, self.ball, grav, rc1, rc2]:
            system.add(el)
        system.assemble()
        self.solver = ScipyIVP(system, 5, 1 / self.fps)
        self.solver.t0 = 0

    def solve_step(self):
        # integration time
        solver = self.solver
        solver.t1 = solver.t0 + solver.dt
        solver.t_eval = np.array([solver.t0, solver.t1])
        solver.frac = solver.dt / 101
        sol = solver.solve()
        solver.t0 += solver.dt
        solver.x0 = np.concatenate([sol.q[-1], sol.u[-1]])
        return sol


# scipy ivp
M_inv = lambda alpha: np.array(
    [
        [m_ball * l2**2, -m_ball * l2 * np.cos(alpha)],
        [-m_ball * l2 * np.cos(alpha), m_cart + m_ball],
    ]
    / (m_cart + m_ball * np.sin(alpha) ** 2)
    / (m_ball * l2**2)
)
h = lambda alpha, dalpha: m_ball * l2 * np.sin(alpha) * np.array([dalpha**2, -g_accel])


class SimulatorNode(Node):

    def __init__(self):
        super().__init__("simulator")
        # subscription
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Forcing, "forcing", self.callback_forcing, qos_profile)
        # model
        # self.cart_ball = CartBall()
        # publisher
        self.publisher = self.create_publisher(CartBallState, "cart_ball_state", 10)
        self.timer = self.create_timer(1 / self.cart_ball.fps, self.timer_callback)
        self.__la = 0.0
        self.__t0 = 0.0
        self.__dt = 1 / self.cart_ball.fps
        self.__y0 = np.array([0, alpha0, 0, 0])

    def callback_forcing(self, msg_forcing):
        self.__la = msg_forcing.la

    def timer_callback(self):
        # # step simulation
        # self.cart_ball.forcing.force = lambda t: np.array([self.__la, 0, 0], dtype=np.float64)
        # sol = self.cart_ball.solve_step()
        # cart = self.cart_ball.cart
        # ball = self.cart_ball.ball
        # self.cart_ball.forcing.force = lambda t: np.zeros(3, dtype=np.float64)
        # ti, qi, ui = sol.t[-1], sol.q[-1], sol.u[-1]
        # # update state
        # message = CartBallState()
        # message.la = self.cart_ball.forcing.force(ti)
        # message.r_os_cart = cart.r_OP(
        #     ti, qi[cart.qDOF]
        # )
        # message.r_os_ball = ball.r_OP(
        #     ti, qi[ball.qDOF]
        # )
        # message.v_s_cart = cart.v_P(
        #     ti, qi[cart.qDOF], ui[cart.uDOF]
        # )
        # message.v_s_ball = ball.r_OP(
        #     ti, qi[ball.qDOF], ui[ball.uDOF]
        # )
        # self.publisher.publish(message)
        # step simulation
        def fun(t, y):
            x, alpha, dx, dalpha = y
            dy = np.empty_like(y)
            dy[:2] = y[2:]
            dy[2:] = M_inv(alpha) @ (h(alpha, dalpha) + np.array([self.__la, 0]))
            return dy

        t0 = self.__t0
        t1 = t0 + self.__dt
        t_eval = [t0, t1]
        s = solve_ivp(
            fun,
            method="Radau",
            t_span=[t0, t1],
            y0=self.__y0,
            t_eval=t_eval,
        )
        self.__y0 = s.y[:, -1]
        x_ivp, alpha_ivp, dx_ivp, dalpha_ivp = s.y[:, -1]
        # update state
        message = CartBallState()
        message.la = self.__la
        message.r_os_cart = np.array([x_ivp, 0.0, 0.0])
        message.r_os_ball = [
            x_ivp + l2 * np.sin(alpha_ivp),
            -l2 * np.cos(alpha_ivp),
            0.0,
        ]
        message.v_s_cart = [dx_ivp, 0.0, 0.0]
        message.v_s_ball = [
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
