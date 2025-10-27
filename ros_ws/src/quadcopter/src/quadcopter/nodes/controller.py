import numpy as np
from scipy.spatial.transform import Rotation as R
from cardillo.math import  quat2axis_angle

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy

from my_interfaces.msg import QuadCopterState, QuadMotorForce



mass = 1.0
g_accel = 9.81
B_Theta_C = np.diag([1.0, 1.0, 3.0])
L = 0.1  # distance from center to motor
H = 0.03  # height of quadcopter body
Km = 0.01  # quadmotor moment/force ratio
input_map = np.array([[1, 1, 1, 1], [-L, L, L, -L], [-L, -L, L, L], [Km, -Km, Km, -Km]])
input_map_inv = np.linalg.inv(input_map)

# controller gain
lam = 1
loop_rate = 20
Kd_r, Kp_r = 2 * lam, lam**2
Kd_p, Kp_p = Kd_r * loop_rate, Kp_r * loop_rate**2


r_OP0 = np.array([0.0, 0.0, 1.0])
def traj(t):
    t_trans = 3
    t_trans2 = 10
    if t <= t_trans:
        r0 = r_OP0
        r1 = r_OP0 + np.array([1, 0, 1])
        s = min(1, t/t_trans)
        r_OP = r0 + (r1 - r0) * (10 * s** 3 - 15 * s ** 4 + 6 * s **5)
        v_P = (r1 - r0) / t_trans * (30 * s** 2 - 60 * s ** 3 + 30 * s **4)
        a_P = (r1 - r0) / (t_trans ** 2) * (60 * s - 180 * s ** 2 + 120 * s **3)
    elif t - t_trans <= t_trans2:
        p0 = 0
        p1 = np.pi * 2
        s = min(1, (t-t_trans)/t_trans2)
        phi = p0 + (p1 - p0) * (10 * s** 3 - 15 * s ** 4 + 6 * s **5)
        phi_dot = (p1 - p0) / t_trans2 * (30 * s** 2 - 60 * s ** 3 + 30 * s **4)
        phi_ddot = (p1 - p0) / (t_trans2 ** 2) * (60 * s - 180 * s ** 2 + 120 * s **3)
        r_OP = r_OP0 + np.array([np.cos(phi), np.sin(phi), 1])
        v_P = np.array([-np.sin(phi), np.cos(phi), 0]) * phi_dot
        a_P = np.array([-np.sin(phi), np.cos(phi), 0]) * phi_ddot + np.array([-np.cos(phi), -np.sin(phi), 0]) * phi_dot**2
    else:
        r_OP = r_OP0 + np.array([1, 0, 1])
        v_P = r_OP0 * 0
        a_P = r_OP0 * 0
        
    return r_OP, v_P, a_P


def position_control(alpha, r_OP, v_P, r_OP_des, v_P_des, a_P_des):
    a_Px, a_Py, a_Pz = a_P_des + Kd_r * (v_P_des - v_P) + Kp_r * (r_OP_des - r_OP)
    B_la_sum = mass * g_accel + mass * a_Pz
    betad = 1 / g_accel * (a_Px * np.cos(alpha) + a_Py * np.sin(alpha))
    gammad = 1 / g_accel * (a_Px * np.sin(alpha) - a_Py * np.cos(alpha))
    return B_la_sum, betad, gammad


def pose_control(angles, angles_dot, anglesd, anglesd_dot, anglesd_ddot):
    tau = (
        1
        / np.diagonal(B_Theta_C)
        * (anglesd_ddot + Kd_p * (anglesd_dot - angles_dot) + Kp_p * (anglesd - angles))
    )
    return tau

class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller")
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            QuadCopterState, "quad_copter_state", self.callback_quad_copter_state, qos_profile
        )
        self.publisher = self.create_publisher(QuadMotorForce, "quad_motor_forces", 10)
        self.active_lqr = False

    def callback_quad_copter_state(self, msg):
        t, q, u = msg.t, msg.q, msg.u
        # trajectory
        r_OP_des, v_P_des, a_P_des = traj(t)
        # quad copter states
        r_OP, v_P = q[:3], u[:3]
        angles = R.from_rotvec(quat2axis_angle(q[3:])).as_euler("zyx", degrees=False)[
            ::-1
        ]
        gamma, beta, alpha = angles
        angles_dot = (
            np.array(
                [
                    [1, np.tan(beta) * np.sin(gamma), np.tan(beta) * np.cos(gamma)],
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma) / np.cos(beta), np.cos(gamma) / np.cos(beta)],
                ]
            )
            @ u[3:]
        )
        # position control
        B_la_sum, betad, gammad = position_control(
            alpha, r_OP, v_P, r_OP_des, v_P_des, a_P_des
        )
        anglesd = np.array([gammad, betad, 0])
        anglesd_dot = np.zeros_like(anglesd)
        anglesd_ddot = np.zeros_like(anglesd)
        # pose control
        B_tau = pose_control(angles, angles_dot, anglesd, anglesd_dot, anglesd_ddot)
        # TODO: map to velocities is more realistic
        las = input_map_inv @ np.hstack((B_la_sum, B_tau))
    
        msg = QuadMotorForce()
        msg.las = las
        self.publisher.publish(msg)


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
