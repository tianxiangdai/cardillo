import numpy as np
from collections import namedtuple
import meshio
from xml.dom import minidom
from pathlib import Path

from cardillo.solver import Solution


class Export:
    def __init__(
        self,
        path: Path,
        folder_name: str,
        overwrite: bool,
        fps: float,
        solution: Solution,
        # system = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.folder = self._create_vtk_folder(folder_name, overwrite)
        self.fps = fps

        # self.system = system
        self._prepare_data(solution)

        self.root = minidom.Document()

        self.vtk_file = self.root.createElement("VTKFile")
        self.vtk_file.setAttribute("type", "Collection")
        self.root.appendChild(self.vtk_file)

        self.collection = self.root.createElement("Collection")
        self.vtk_file.appendChild(self.collection)

    Data = namedtuple("Data", ["points", "cells", "point_data", "cell_data"])

    # helper functions
    def _unique_file_name(self, file_name):
        file_name_ = file_name
        i = 1
        while (self.path / f"{file_name_}.pvd").exists():
            file_name_ = f"{file_name}{i}"
            i += 1
        return file_name_

    def _write_time_step_and_name(self, t, file):
        # write time step and file name in pvd file
        dataset = self.root.createElement("DataSet")
        dataset.setAttribute("timestep", f"{t:0.6f}")
        dataset.setAttribute("file", file.name)
        self.collection.appendChild(dataset)

    def _write_pvd_file(self, path):
        xml_str = self.root.toprettyxml(indent="\t")
        with (path).open("w") as f:
            f.write(xml_str)

    def _prepare_data(self, sol):
        frames = len(sol.t)
        # target_frames = min(len(t), 100)
        animation_time_ = sol.t[-1] - sol.t[0]
        target_frames = int(animation_time_ * self.fps)
        frac = int(frames / target_frames)

        frames = target_frames
        t = sol.t[::frac]
        q = sol.q[::frac]
        u = sol.u[::frac]
        if hasattr(sol, "u_dot"):
            u_dot = sol.u_dot[::frac]
        else:
            u_dot = None
        la_g = sol.la_g[::frac]
        la_gamma = sol.la_gamma[::frac]
        if hasattr(sol, "P_N"):
            P_N = sol.P_N[::frac]
        else:
            P_N = None
        if hasattr(sol, "P_F"):
            P_F = sol.P_F[::frac]
        else:
            P_F = None

        # TODO default values + not None values of solution object
        self.solution = Solution(
            t=t, q=q, u=u, u_dot=u_dot, la_g=la_g, la_gamma=la_gamma, P_N=P_N, P_F=P_F
        )

    def _create_vtk_folder(self, folder_name: str, overwrite: bool):
        path = self.path / folder_name
        i = 0
        if not overwrite:
            while path.exists():
                path = self.path / str(folder_name + f"_{i}")
                i += 1
        # TODO: delete existing files
        path.mkdir(parents=True, exist_ok=overwrite)
        self.path = path

    def _export_list(self, sol_i):
        points, cells, point_data, cell_data = [], [], {}, {}
        l = 0
        for contr in self.contr_list:
            p, c, p_data, c_data = contr.export(sol_i)
            l = len(points)
            points.extend(p)
            cells.extend([(el[0], [[el[1][0][0] + l]]) for el in c])
            if c_data is not None:
                for key in c_data.keys():
                    if not key in cell_data.keys():
                        cell_data[key] = c_data[key]
                    else:
                        cell_data[key].extend(c_data[key])
            if p_data is not None:
                for key in p_data.keys():
                    if not key in point_data.keys():
                        point_data[key] = p_data[key]
                    else:
                        point_data[key].extend(p_data[key])

        return points, cells, point_data, cell_data

    def export_contr(self, contr):
        # export one contr
        if not isinstance(contr, (list, tuple, np.ndarray)):
            contr_name = contr.__class__.__name__
            export = contr.export
        else:
            # assume list of same contr types
            contr_name = contr[0].__class__.__name__
            self.contr_list = contr
            export = self._export_list

        file_name = self._unique_file_name(contr_name)
        for i, sol_i in enumerate(self.solution):
            file_i = self.path / f"{file_name}_{i}.vtu"
            self._write_time_step_and_name(sol_i.t, file_i)

            points, cells, point_data, cell_data = export(sol_i)

            meshio.write_points_cells(
                filename=file_i,
                points=points,
                cells=cells,
                point_data=point_data,
                cell_data=cell_data,
                binary=False,
            )
        self._write_pvd_file(self.path / f"{file_name}.pvd")

    # def _exportTranslationalForce(self, sol_i):
    #     points, cells = [], []
    #     offset = 0
    #     for force in self.contr:
    #         if isinstance(force, TranslationalForceTri):
    #             points.append(force.r_OBk(sol_i.t, sol_i.q[force.qDOF], 0))
    #             points.append(points[-1] + force.median(sol_i.t, sol_i.q[force.qDOF]))
    #             new_con = [np.array([offset, offset+1])]
    #         elif isinstance(force, TranslationalForce_n):
    #             for i, _ in enumerate(force.subsystems):
    #                 points.append(force.r_OBk(sol_i.t, sol_i.q[force.qDOF], i))
    #             new_con = [np.array(tup)+np.full((2), offset) for tup in force.connectivity]
    #         else:
    #             points.append(force.r_OP1(sol_i.t, sol_i.q[force.qDOF]))
    #             points.append(force.r_OP2(sol_i.t, sol_i.q[force.qDOF]))
    #             new_con = [np.array([offset, offset+1])]
    #         offset = len(points)
    #         cells.append(("line", np.array(new_con)))
    #     points = np.array(points)
    #     point_data = cell_data = None # TODO add cell_data

    #     return points, cells, point_data, cell_data

    # def _exportConvexBody(self, sol_i):
    #     points, cells = [], []
    #     offset = 0
    #     for convex_body in self.contr:
    #         cells_connectivity = offset + convex_body.mesh.simplices
    #         normals = convex_body.A
    #         for point in convex_body.mesh.points:
    #             points.append(convex_body.r_OP(sol_i.t, sol_i.q[convex_body.qDOF], K_r_SP=point))

    #         cells.append(("triangle", cells_connectivity))

    #         normals = np.array(
    #             [
    #                 convex_body.A_IK(sol_i.t, sol_i.q[convex_body.qDOF]) @ normals[j, :]
    #                 for j in range(normals.shape[0])
    #             ]
    #         )

    #         offset = len(points)
    #     points = np.array(points)
    #     point_data = cell_data = None

    #     return points, cells, point_data, cell_data
