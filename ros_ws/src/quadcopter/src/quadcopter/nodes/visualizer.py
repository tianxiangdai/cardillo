import vtk
from time import perf_counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy


from my_interfaces.msg import QuadCopterState

max_fps = 60
L = 0.1  # distance from center to motor
H = 0.03  # height of quadcopter body


class VisualizerNode(Node):

    def __init__(self):
        super().__init__("visualizer")
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            QuadCopterState, "quad_copter_state", self.callback_quad_copter_state, qos_profile
        )
        # body
        box = vtk.vtkCubeSource()
        box.SetXLength(2*L)
        box.SetYLength(2*L)
        box.SetZLength(H)

        self.H_IB_body = vtk.vtkMatrix4x4()
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB_body)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(box.GetOutputPort())
        tf_filter.SetTransform(_H_IB)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor_quadcopter = vtk.vtkActor()
        actor_quadcopter.SetMapper(mapper)


        # renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor_quadcopter)
        ren.SetBackground(vtk.vtkNamedColors().GetColor3d("DarkGreen"))

        self.win = vtk.vtkRenderWindow()
        self.win.SetWindowName("")
        self.win.AddRenderer(ren)
        self.win.MakeRenderWindowInteractor()
        self.win.SetSize(800, 600)
        self.interactor = self.win.GetInteractor()
        self.interactor.AddObserver(
            vtk.vtkCommand.ExitEvent, self.__handle_window_closed
        )
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(ren)
        self.cam_widget.On()
        self.time = -1

    def __handle_window_closed(self, inter, event):
        rclpy.shutdown()

    def callback_quad_copter_state(self, msg):
        t, q, u = msg.t, msg.q, msg.u
        ctime = perf_counter()
        if ctime - self.time < 1 / max_fps:
            return
        self.time = ctime
        for i in range(3):
            self.H_IB_body.SetElement(i, 3, q[i])
        self.H_IB_body.Modified()
        self.win.Render()
        self.interactor.ProcessEvents()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
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
