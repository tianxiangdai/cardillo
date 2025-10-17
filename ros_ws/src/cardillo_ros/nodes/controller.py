import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy

from my_interfaces.msg import CartBallState, Forcing


class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller")
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            CartBallState, "cart_ball_state", self.callback_cart_ball_state, qos_profile
        )
        self.publisher = self.create_publisher(Forcing, "forcing", 10)

    def callback_cart_ball_state(self, msg_cart_ball_state):
        m_cart = 1
        m_ball = 1
        g_accel = 9.81
        l2 = 0.1
        # la = msg_cart_ball_state.la
        r_OS_cart = msg_cart_ball_state.r_os_cart
        r_OS_ball = msg_cart_ball_state.r_os_ball
        dr = r_OS_ball - r_OS_cart
        v_S_cart = msg_cart_ball_state.v_s_cart
        v_S_ball = msg_cart_ball_state.v_s_ball
        dv = v_S_ball - v_S_cart
        x, dx = r_OS_cart[0], v_S_cart[0]
        if dr[1] < 0:
            alpha = np.arcsin(dr[0] / l2)
        else:
            alpha = np.pi - np.arcsin(dr[0] / l2)
        dalpha = np.linalg.norm(dv) / l2 * np.sign(np.cross([0, 0, 1], dr) @ dv)

        E = 0.5 * (
            (m_cart + m_ball) - m_ball * np.cos(alpha) ** 2
        ) * l2**2 * dalpha**2 - (m_cart + m_ball) * g_accel * l2 * np.cos(alpha)
        dE = E - (m_cart + m_ball) * g_accel * l2
        if dx == 0:
            la = 0.0
        else:
            # la = - np.arctan(dE * dx * 1000) *2 /np.pi * 10
            la = np.arctan(dE * l2 * np.cos(alpha) * dalpha * 10) * 2 / np.pi * 10.0
        print(E)
        message = Forcing()
        message.la = la
        self.publisher.publish(message)
        # print(dE)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
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
