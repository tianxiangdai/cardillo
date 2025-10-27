import numpy as np

from cardillo import System
from cardillo.discrete import RigidBody, Box
from cardillo.forces import Force
from cardillo.solver import ScipyIVP
from cardillo.math import axis_angle2quat

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy

from my_interfaces.msg import QuadCopterState, QuadMotorForce


fps = 1000

mass = 1.0
g_accel = 9.81
B_Theta_C = np.diag([1.0, 1.0, 3.0])
L = 0.1  # distance from center to motor
H = 0.03  # height of quadcopter body
Km = 0.01  # quadmotor moment/force ratio
input_map = np.array([[1, 1, 1, 1], [-L, L, L, -L], [-L, -L, L, L], [Km, -Km, Km, -Km]])

r_OP0 = np.array([0.0, 0.0, 1.0])
p0 = axis_angle2quat(np.array([1.0, 1.0, 1.0]), np.deg2rad(0))
q0 = np.hstack((r_OP0, p0))


class QuadMotor:
    def __init__(self, subsystem, xi=np.zeros(3), name="quad_motor"):
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

        B_r_CPs = np.array(
            [[L, -L, H / 2], [L, L, H / 2], [-L, L, H / 2], [-L, -L, H / 2]]
        )

        self.A_IB = lambda t, q: subsystem.A_IB(t, q, xi=xi)
        self.A_IB_q = lambda t, q: subsystem.A_IB_q(t, q, xi=xi)
        self.r_OP = lambda t, q, k: subsystem.r_OP(t, q, xi=xi, B_r_CP=B_r_CPs[k])
        self.J_P = lambda t, q, k: subsystem.J_P(t, q, xi=xi, B_r_CP=B_r_CPs[k])
        self.J_P_q = lambda t, q, k: subsystem.J_P_q(t, q, xi=xi, B_r_CP=B_r_CPs[k])
        self.las = np.zeros(4)


    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        # body fixed moment
        # self.las = self.control_input(t, q, u)
        B_tau = input_map[1:] @ self.las
        h = B_tau @ self.B_J_R(t, q)
        for i in range(4):
            h += (self.A_IB(t, q) @ np.array([0, 0, self.las[i]])) @ self.J_P(t, q, i)
        return h

    def h_q(self, t, q, u):
        # self.las = self.control_input(t, q, u)
        B_tau = input_map[1:] @ self.las
        h_q = np.einsum("i,ijk->jk", B_tau, self.B_J_R_q(t, q))
        for i in range(4):
            B_la = np.array([0, 0, self.las[i]])
            h += np.einsum(
                "ijk,j,il->lk", self.A_IB_q(t, q), B_la, self.J_P(t, q, i)
            ) + np.einsum("i,ijk->jk", self.A_IB(t, q) @ B_la, self.J_P_q(t, q, i))
        return h_q


class Quadcopter:
    def __init__(self):
        self.fps = fps

        self.system = System()       
    
        self.body = Box(RigidBody)(
            dimensions=[2 * L, 2 * L, H],
            mass=mass,
            B_Theta_C=B_Theta_C,
            q0=q0,
            name="quadcopter",
        )
        gravity = Force(np.array([0.0, 0.0, -mass * g_accel]), self.body, name="gravity")

        self.quad_motor = QuadMotor(self.body)
        
        self.system.add(self.body)
        self.system.add(gravity)
        self.system.add(self.quad_motor)
        self.system.assemble()
        
        self.solver = ScipyIVP(self.system, 1 / self.fps, 1 / self.fps)
        assert len(self.solver.t_eval) == 2

    def set_force(self, las):
        self.quad_motor.las = las

    def set_solver(self, q0, u0):
        self.system.set_new_initial_state(q0, u0)
        self.solver.x0 = np.concatenate([q0, u0])

    def solve(self):
        return self.solver.solve()


class SimulatorNode(Node):

    def __init__(self):
        super().__init__("simulator")
        # subscription
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(QuadMotorForce, "quad_motor_forces", self.callback_motor_force, qos_profile)
        # model
        self.quadcopter = Quadcopter()
        # publisher
        self.publisher = self.create_publisher(QuadCopterState, "quad_copter_state", 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.__la = np.zeros(4, np.float64)
        self.__nt = 0
        
    def callback_motor_force(self, msg):
        self.__la = msg.las

    def timer_callback(self):
        # step simulation
        body = self.quadcopter.body
        self.quadcopter.set_force(self.__la)
        sol = self.quadcopter.solve()
        ti, qi, ui = sol.t[-1], sol.q[-1], sol.u[-1]
        self.quadcopter.set_solver(qi, ui)
        # update state
        self.__nt += 1
        msg = QuadCopterState()
        msg.t = self.__nt * 1 / fps
        msg.q = qi[body.qDOF]
        msg.u = ui[body.uDOF]
        self.publisher.publish(msg)

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
