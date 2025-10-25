import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy

from my_interfaces.msg import CartPoleState, Forcing


m_cart = 1
m_pole = 1
g_accel = 9.81
l2 = 0.1


class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller")
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            CartPoleState, "cartpole_state", self.callback_cartpole_state, qos_profile
        )
        self.publisher = self.create_publisher(Forcing, "forcing", 10)
        self.active_lqr = False

    def callback_cartpole_state(self, msg_state):
        # la = msg_cartpole_state.la
        r_OS_cart = msg_state.r_os_cart
        r_OS_pole = msg_state.r_os_pole
        dr = r_OS_pole - r_OS_cart
        v_S_cart = msg_state.v_s_cart
        v_S_pole = msg_state.v_s_pole
        dv = v_S_pole - v_S_cart
        x, dx = r_OS_cart[0], v_S_cart[0]
        if dr[1] < 0:
            alpha = np.arcsin(dr[0] / l2)
        else:
            alpha = np.pi - np.arcsin(dr[0] / l2)
        dalpha = np.linalg.norm(dv) / l2 * np.sign(np.cross([0, 0, 1], dr) @ dv)
        if not self.active_lqr:
            if np.abs(alpha - np.pi) < np.deg2rad(5):
                self.active_lqr = True
                self.x_goal = np.array([x, np.pi, 0, 0])
                print("active lqr: ", self.x_goal)
            elif np.abs(alpha + np.pi) < np.deg2rad(5):
                self.active_lqr = True
                self.x_goal = np.array([x, -np.pi, 0, 0])
                print("active lqr: ", self.x_goal)

        if self.active_lqr:
            K_lqr = np.array([0.0000, 100.2027, 0.0148, 4.4752])
            la = K_lqr @ (self.x_goal - np.array([x, alpha, dx, dalpha]))
        else:
            E = 0.5 * (
                (m_cart + m_pole) - m_pole * np.cos(alpha) ** 2
            ) * l2**2 * dalpha**2 - (m_cart + m_pole) * g_accel * l2 * np.cos(alpha)
            dE = E - (m_cart + m_pole) * g_accel * l2
            if dx == 0:
                la = 0.0
            else:
                # la = - np.arctan(dE * dx * 1000) *2 /np.pi * 10
                la = np.arctan(dE * l2 * np.cos(alpha) * dalpha * 10) * 2 / np.pi * 10.0
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
