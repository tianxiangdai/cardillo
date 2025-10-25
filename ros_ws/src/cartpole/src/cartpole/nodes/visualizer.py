import vtk
from time import perf_counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy


from my_interfaces.msg import CartPoleState

max_fps = 60


class VisualizerNode(Node):

    def __init__(self):
        super().__init__("visualizer")
        qos_profile = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            CartPoleState, "cartpole_state", self.callback_cartpole_state, qos_profile
        )
        # cart
        box = vtk.vtkCubeSource()
        box.SetXLength(0.03)
        box.SetYLength(0.01)
        box.SetZLength(0.01)

        self.H_IB_cart = vtk.vtkMatrix4x4()
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB_cart)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(box.GetOutputPort())
        tf_filter.SetTransform(_H_IB)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor_cart = vtk.vtkActor()
        actor_cart.SetMapper(mapper)

        # pole
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.01)
        sphere.SetPhiResolution(20)
        sphere.SetThetaResolution(20)

        self.H_IB_pole = vtk.vtkMatrix4x4()
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB_pole)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(sphere.GetOutputPort())
        tf_filter.SetTransform(_H_IB)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor_pole = vtk.vtkActor()
        actor_pole.SetMapper(mapper)

        # line
        self.line = vtk.vtkLineSource()
        self.line.SetPoint1([0, 0, 0])
        self.line.SetPoint2([0, 0, 0])
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.line.GetOutputPort())
        actor_line = vtk.vtkActor()
        actor_line.SetMapper(mapper)

        # renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor_cart)
        ren.AddActor(actor_pole)
        ren.AddActor(actor_line)
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

    def callback_cartpole_state(self, msg_cartpole_state):
        # la = msg_cartpole_state.la
        ctime = perf_counter()
        if ctime - self.time < 1 / max_fps:
            return
        self.time = ctime
        r_OS_cart = msg_cartpole_state.r_os_cart
        r_OS_pole = msg_cartpole_state.r_os_pole
        self.line.SetPoint1(*r_OS_cart)
        self.line.SetPoint2(*r_OS_pole)
        for i in range(3):
            self.H_IB_cart.SetElement(i, 3, r_OS_cart[i])
            self.H_IB_pole.SetElement(i, 3, r_OS_pole[i])
        self.H_IB_cart.Modified()
        self.H_IB_pole.Modified()
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
