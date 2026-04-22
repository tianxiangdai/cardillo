import numpy as np
from abc import ABC, abstractmethod
from time import perf_counter, sleep
import threading

import vtk
from cardillo.interactions.n_point_interaction import nPointInteraction
from cardillo.rods import CircularCrossSection, RectangularCrossSection
from cardillo.math_jax import Exp_SO3_quat_batch

from cardillo.rods._base import CosseratRod_PetrovGalerkin
from cardillo.rods.discreteRod import DiscreteRod
from cardillo.solver.solution import Solution


class _VisualTwinBase(ABC):
    def __init__(self, contr, xi=None):
        self.xi = xi
        if isinstance(contr, DiscreteRod):
            xi = 0 if xi is None else xi
            s = contr.get_marker(xi)
            if hasattr(contr, "qDOF"):
                num = contr.element_number(xi)
                s.t0 = contr.t0
                s.q0 = contr.q0[contr.elDOF[num]]
                s.qDOF = contr.qDOF[contr.elDOF[num]]
                s.uDOF = contr.uDOF[contr.elDOF_u[num]]
            contr = s
        self.contr = contr
        self.actors = []
        if not hasattr(contr, "visual_twins"):
            contr.visual_twins = [self]
        else:
            contr.visual_twins.append(self)

    @abstractmethod
    def update_state(self, sol_i):
        pass


class VisualDiscreteRod(_VisualTwinBase):
    def __init__(
        self,
        rod,
        radius,
        subdivision=3,
        color=(82, 108, 164),
        opacity=1,
    ):
        super().__init__(rod)
        self.rod = rod
        nelement_visual = rod.nelement
        self.nelement_visual = rod.nelement
        self.nnode_visual = rod.nnode
        self.xi_node = rod.xi_node

        if isinstance(rod.cross_section, CircularCrossSection):
            weights = [
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
            ]
            degrees = [2, 2, 1]
            ctype = vtk.VTK_BEZIER_WEDGE
        # elif isinstance(rod.cross_section, RectangularCrossSection):
        #     npts = 16
        #     weights = [1] * 16
        #     degrees = [1, 1, 3]
        #     ctype = vtk.VTK_BEZIER_HEXAHEDRON
        else:
            raise NotImplementedError

        ugrid = vtk.vtkUnstructuredGrid()

        # points
        self.body_points = vtk.vtkPoints()
        self.body_points.SetNumberOfPoints(6 * (nelement_visual + 1))
        ugrid.SetPoints(self.body_points)

        # cells
        ugrid.Allocate(nelement_visual)
        for i in range(nelement_visual):
            ugrid.InsertNextCell(
                ctype,
                12,
                list(range(i * 6, i * 6 + 3))
                + list(range((i + 1) * 6, (i + 1) * 6 + 3))
                + list(range(i * 6 + 3, (i + 1) * 6))
                + list(range((i + 1) * 6 + 3, (i + 2) * 6)),
            )

        # point data
        self.pdata = ugrid.GetPointData()
        value = weights * (nelement_visual + 1)
        parray = vtk.vtkDoubleArray()
        parray.SetName("RationalWeights")
        parray.SetNumberOfTuples(6)
        parray.SetNumberOfComponents(1)
        for i, vi in enumerate(value):
            parray.InsertTuple(i, [vi])
        self.pdata.SetRationalWeights(parray)

        # cell data
        self.cdata = ugrid.GetCellData()
        carray = vtk.vtkIntArray()
        carray.SetName("HigherOrderDegrees")
        carray.SetNumberOfTuples(nelement_visual)
        carray.SetNumberOfComponents(3)
        for i in range(nelement_visual):
            carray.InsertTuple(i, degrees)
        self.cdata.SetHigherOrderDegrees(carray)

        filter = vtk.vtkDataSetSurfaceFilter()
        filter.SetInputData(ugrid)
        filter.SetNonlinearSubdivisionLevel(subdivision)

        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

        # control points on circle
        self.control_pts_circle = np.empty((self.nnode_visual, 6, 3))
        for i in range(self.nnode_visual):
            if callable(radius):
                _radius = radius(self.xi_node[i])
            else:
                _radius = radius
            phis = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
            xys1 = (
                np.stack([np.zeros_like(phis), np.cos(phis), np.sin(phis)], axis=1)
                * _radius
            )
            # control points out of circle
            phis2 = phis + (np.pi / 3.0)
            xys2 = np.stack(
                [np.zeros_like(phis), np.cos(phis2), np.sin(phis2)], axis=1
            ) * (2.0 * _radius)
            self.control_pts_circle[i] = np.concatenate([xys1, xys2], axis=0)

    def update_state(self, sol_i):
        nnodes = self.nnode_visual
        control_pts = np.empty((nnodes * 6, 3), dtype=np.float64)
        qs = sol_i.q[self.rod.qDOF].reshape((nnodes, 7))
        r_OCs = qs[:, :3]
        A_IBs = Exp_SO3_quat_batch(qs[:, 3:], True)
        for i, r_OC, A_IB in zip(range(nnodes), r_OCs, A_IBs):
            pts = r_OC + self.control_pts_circle[i] @ A_IB.T
            base = i * 6
            control_pts[base : base + 6] = pts
        body_points = self.body_points
        set_point = body_points.SetPoint
        for i, p in enumerate(control_pts):
            set_point(i, p)
        body_points.Modified()


class _VisualvtkSource(_VisualTwinBase):
    def __init__(
        self,
        contr,
        xi,
    ):
        super().__init__(contr, xi)
        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            self.N, self.N_xi = self.contr.basis_functions_r(xi)

    def add_vtk_source(
        self,
        source,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):

        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB)
        _H_IM = vtk.vtkTransform()
        _H_IM.PostMultiply()
        _H_IM.SetMatrix(H_BM.flatten())
        _H_IM.Concatenate(_H_IB)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(source.GetOutputPort())
        tf_filter.SetTransform(_H_IM)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update_state(self, sol_i):
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        xi = self.xi
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            qe = q[self.contr.local_qDOF_P(xi)]
            r_OP, A_IB, _, _ = self.contr._eval(qe, xi, self.N, self.N_xi)
        else:
            A_IB = self.contr.A_IB(t, q, xi)
            r_OP = self.contr.r_OP(t, q, xi)
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])


class VisualArUco(_VisualTwinBase):
    def __init__(
        self,
        contr,
        xi=None,
        mk_size=0.04,
        mk_dis=0.045,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        opacity=1,
    ):
        super().__init__(contr, xi)
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            self.N, self.N_xi = self.contr.basis_functions_r(xi)
        from cv2 import aruco

        n_row = 2
        n_col = 2
        x0 = -mk_size / 2 - mk_dis / 2
        y0 = -x0
        h0 = 1e-4
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        quads_black = vtk.vtkCellArray()
        quads_white = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        for row in range(n_row):
            for col in range(n_col):
                id = row * n_col + col
                qrcode = aruco_dict.generateImageMarker(id, aruco_dict.markerSize + 2)
                bit_size = mk_size / (aruco_dict.markerSize + 2)

                # Create a triangle
                n_bits = qrcode.shape[0]
                for i in range(n_bits + 1):
                    for j in range(n_bits + 1):
                        points.InsertNextPoint(
                            x0 + col * mk_dis + j * bit_size,
                            y0 - row * mk_dis - i * bit_size,
                            h0,
                        )

                for i in range(n_bits):
                    for j in range(n_bits):
                        quad = vtk.vtkQuad()
                        quad.GetPointIds().SetId(
                            0, id * (n_bits + 1) ** 2 + i * (n_bits + 1) + j
                        )
                        quad.GetPointIds().SetId(
                            1, id * (n_bits + 1) ** 2 + (i + 1) * (n_bits + 1) + j
                        )
                        quad.GetPointIds().SetId(
                            2, id * (n_bits + 1) ** 2 + (i + 1) * (n_bits + 1) + j + 1
                        )
                        quad.GetPointIds().SetId(
                            3, id * (n_bits + 1) ** 2 + i * (n_bits + 1) + j + 1
                        )
                        if qrcode[i, j] == 0:
                            quads_black.InsertNextCell(quad)
                        else:
                            quads_white.InsertNextCell(quad)

        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB)
        _H_IM = vtk.vtkTransform()
        _H_IM.PostMultiply()
        _H_IM.SetMatrix(H_BM.flatten())
        _H_IM.Concatenate(_H_IB)

        # qrcode
        for triangles, color in zip(
            [quads_black, quads_white], [(0, 0, 0), (255, 255, 255)]
        ):
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(triangles)

            filter = vtk.vtkTransformPolyDataFilter()
            filter.SetInputData(polydata)
            filter.SetTransform(_H_IM)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetColor([c / 255 for c in color])
            actor.GetProperty().SetOpacity(opacity)
            actor.SetMapper(mapper)
            self.actors.append(actor)
            # subsystem.appendfilter.AddInputConnection(filter.GetOutputPort())

    def update_state(self, sol_i):
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        xi = self.xi
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            qe = q[self.contr.local_qDOF_P(xi)]
            r_OP, A_IB, _, _ = self.contr._eval(qe, xi, self.N, self.N_xi)
        else:
            A_IB = self.contr.A_IB(t, q, xi)
            r_OP = self.contr.r_OP(t, q, xi)
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])


class VisualCylinder(_VisualvtkSource):
    def __init__(
        self,
        contr,
        radius,
        height,
        xi=None,
        resolution=30,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkCylinderSource()
        source.SetRadius(radius)
        source.SetHeight(height)
        source.SetResolution(resolution)
        A_BM = A_BM @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T
        self.add_vtk_source(source, A_BM, B_r_CP, color, opacity)


class VisualSTL(_VisualvtkSource):
    def __init__(
        self,
        contr,
        stl_file,
        xi=None,
        scale=1e-3,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkSTLReader()
        source.SetFileName(stl_file)
        source.Update()
        self.add_vtk_source(source, A_BM * scale, B_r_CP, color, opacity)


class VisualCoordSystem(_VisualvtkSource):
    def __init__(
        self,
        contr,
        length,
        xi=None,
        resolution=30,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkArrowSource()
        source.SetTipResolution(resolution)
        source.SetShaftResolution(resolution)
        for i in range(3):
            if i == 0:
                color = (255, 0, 0)
            elif i == 1:
                A_BM = A_BM @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                color = (0, 255, 0)
            elif i == 2:
                A_BM = A_BM @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                color = (0, 0, 255)
            self.add_vtk_source(source, A_BM * length, B_r_CP, color, opacity)


class VisualTendon(_VisualTwinBase):
    def __init__(
        self, tendon: nPointInteraction, radius=1e-3, color=(255, 255, 255), opacity=1
    ):
        super().__init__(tendon)
        poly_data = vtk.vtkPolyData()
        # points
        npts = 2
        ncon = len(self.contr.connectivity)
        self.vtkpoints = vtk.vtkPoints()
        self.vtkpoints.SetNumberOfPoints(npts * ncon)
        poly_data.SetPoints(self.vtkpoints)

        # cells
        poly_data.Allocate(ncon)
        for i in range(ncon):
            poly_data.InsertNextCell(
                vtk.VTK_LINE, npts, list(range(i * npts, (i + 1) * npts))
            )
        filter = vtk.vtkTubeFilter()
        filter.SetRadius(radius)
        filter.SetInputData(poly_data)
        filter.SetNumberOfSides(50)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(([c / 255 for c in color]))
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update_state(self, sol_i):
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        points = []
        for j, k in self.contr.connectivity:
            points.append(self.contr.r_OPk(t, q, j))
            points.append(self.contr.r_OPk(t, q, k))
        for i, p in enumerate(points):
            self.vtkpoints.SetPoint(i, p)
        self.vtkpoints.Modified()


class Plotter:
    def __init__(self, system, window_size):
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(*window_size)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        self.window.AddRenderer(self.ren)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.window.SetInteractor(self.interactor)

        # ground
        # grid_size = 0.2
        # plane = vtk.vtkPlaneSource()
        # plane.SetOrigin(0, -grid_size, -grid_size)
        # plane.SetPoint1(0, -grid_size, grid_size,)
        # plane.SetPoint2(0, grid_size, -grid_size,)
        # plane.SetXResolution(10)
        # plane.SetYResolution(10)

        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(plane.GetOutputPort())

        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetRepresentationToWireframe()
        # actor.GetProperty().SetColor(0.6, 0.6, 0.6)
        # self.ren.AddActor(actor)

        # camera
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.ren)
        self.cam_widget.On()
        self.camera = self.ren.GetActiveCamera()
        self.camera.ParallelProjectionOff()

        self.window.AddRenderer(self.ren)
        self.__visual_twins = []
        self.system = system
        for contr in system.contributions:
            if hasattr(contr, "visual_twins"):
                for twin in contr.visual_twins:
                    self.__add_visual_twin(twin)
            if hasattr(contr, "_markers"):
                for marker in contr._markers:
                    if hasattr(marker, "visual_twins"):
                        for twin in marker.visual_twins:
                            self.__add_visual_twin(twin)

        self.__do_render = False

        def decorate_step_callback(step_callback):
            def __step_callback(t, q, u):
                r = step_callback(t, q, u)
                if self.__do_render:
                    self.step_render(Solution(self.system, t=t, q=q, u=u))
                return r

            return __step_callback

        system.step_callback = decorate_step_callback(system.step_callback)

        def cbk(interactor, event):
            if interactor.key_code == "q":
                self.hide()

        self.window.SetOffScreenRendering(1)
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, cbk)

    def add_ground(
        self, x0=None, x1=None, y0=None, y1=None, subdivision_x=10, subdivision_y=10
    ):
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(x0, y0, 0)
        plane.SetPoint1(x1, y0, 0)
        plane.SetPoint2(x0, y1, 0)
        plane.SetXResolution(subdivision_x)
        plane.SetYResolution(subdivision_y)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.6, 0.6, 0.6)
        self.ren.AddActor(actor)

    def step_render(self, sol_i):
        for twin in self.__visual_twins:
            twin.update_state(sol_i)
        self.window.Render()
        self.interactor.ProcessEvents()

    def __add_visual_twin(self, visual_twin: _VisualTwinBase):
        if visual_twin not in self.__visual_twins:
            self.__visual_twins.append(visual_twin)
            for actor in visual_twin.actors:
                self.ren.AddActor(actor)
        else:
            raise Exception("visual twin already added!")

    def render_solution(self, solution, repeat=False, play_speed_up=1):
        while True:
            t0_sim = solution.t[0]
            t0_real = perf_counter()
            for i, sol_i in enumerate(solution):
                if i > 0:
                    t_real = perf_counter() - t0_real
                    t_sim = (sol_i.t - t0_sim) / play_speed_up
                    dt = t_real - t_sim
                    if dt > 0.0:
                        continue
                    else:
                        sleep(-dt)
                self.step_render(sol_i)
                if not self.__do_render:
                    return
            if not repeat:
                break
            else:
                sleep(solution.t[1] / play_speed_up)

    def show(self):
        self.window.SetOffScreenRendering(0)
        self.__do_render = True

    def hide(self):
        self.window.SetOffScreenRendering(1)
        self.__do_render = False
