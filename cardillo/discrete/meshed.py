import numpy as np
import trimesh
from vtk import (
    VTK_TRIANGLE,
    VTK_BEZIER_WEDGE,
    VTK_LAGRANGE_HEXAHEDRON,
    VTK_LAGRANGE_TETRAHEDRON,
    VTK_BEZIER_TRIANGLE,
    VTK_BEZIER_QUADRILATERAL,
)


def Meshed(Base):
    """Generate an object (typically with Base `Frame` or `RigidBody`)
    from a given Trimesh object.

    Parameters
    ----------
    Base :  object
        Cardillo object Frame, RigidBody or Pointmass

    Returns
    -------
    out : object
        Meshed version of Base object

    """

    class _Meshed(Base):
        def __init__(
            self,
            mesh_obj,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            scale=1,
            **kwargs,
        ):
            """Generate an object (typically with Base `Frame` or `RigidBody`)
            from a given Trimesh object.

            Parameters
            ----------
            mesh_obj :
                File-like object defining source of mesh or instance of trimesh
                defining the mesh
            Density : float or None
                Mass density for the computation of the inertia properties of the
                mesh. If set to None, user specified mass and B_Theta_C are used.
            B_r_CP : np.ndarray (3,)
                Offset center of mass (C) from (C)TL origin (P) in body fixed K-basis.
            A_BM: np.ndarray (3, 3)
                Tansformation from mesh-fixed basis (M) to body-fixed basis (K).
            scale: float
                Factor scaling the mesh after import.
            kwargs: dict,
                Arguments of parent class (Base) as keyword arguments
            """
            self.B_r_CP = B_r_CP
            self.A_BM = A_BM

            #############################
            # consistency checks for mesh
            #############################
            if isinstance(mesh_obj, trimesh.Trimesh):
                trimesh_obj = mesh_obj

                # primitives are converted to mesh
                if hasattr(trimesh_obj, "to_mesh"):
                    trimesh_mesh = trimesh_obj.to_mesh()
                else:
                    trimesh_mesh = trimesh_obj
            else:
                trimesh_mesh = trimesh.load_mesh(mesh_obj)

            trimesh_mesh.apply_transform(np.diag([scale, scale, scale, 1]))

            # store visual mesh in body fixed basis
            H_KM = np.eye(4)
            H_KM[:3, 3] = B_r_CP
            H_KM[:3, :3] = A_BM
            self.B_visual_mesh = trimesh_mesh.copy().apply_transform(H_KM)

            # vectors (transposed) from (C) to vertices (Qi) represented in body-fixed basis
            self.B_r_CQi_T = self.B_visual_mesh.vertices.view(np.ndarray).T

            # compute inertia quantities of body
            if density is not None:
                # check if mesh represents a valid volume
                if not trimesh_mesh.is_volume:
                    print(
                        "Imported mesh does not represent a volume, i.e. one of the following properties are not fulfilled: watertight, consistent winding, outward facing normals."
                    )
                    # try to fill the wholes
                    trimesh_mesh.fill_holes()
                    if not trimesh_mesh.is_volume:
                        print(
                            "Using mesh that is not a volume. Computed mass and moment of inertia might be unphyical."
                        )
                    else:
                        print("Fixed mesh by filling the holes.")
                # set density and compute properties
                self.B_visual_mesh.density = density
                mass = self.B_visual_mesh.mass
                B_Theta_C = self.B_visual_mesh.moment_inertia

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )

                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(
                    sol_i.t, sol_i.q[self.qDOF]
                )  # TODO: Idea: slicing could be done on global level in Export class. Moreover, solution class should be able to return the slice, e.g., sol_i.get_q_of_body(name).
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                points = (r_OC[:, None] + A_IB @ self.B_r_CQi_T).T

                cells = [(VTK_TRIANGLE, face) for face in self.B_visual_mesh.faces]

            return points, cells, None, None

    return _Meshed


def Box(Base):
    class _Box(Base):
        def __init__(
            self,
            dimensions=np.ones(3),
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            **kwargs,
        ):
            self.dimensions = dimensions
            # compute inertia quantities of body
            if density is not None:
                mass = density * dimensions[0] * dimensions[1] * dimensions[2]
                B_Theta_C = (
                    np.diag(
                        [
                            dimensions[1] ** 2 + dimensions[2] ** 2,
                            dimensions[0] ** 2 + dimensions[2] ** 2,
                            dimensions[0] ** 2 + dimensions[1] ** 2,
                        ]
                    )
                    * mass
                    / 12
                )

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            super().__init__(**kwargs)
            # mesh for visualization
            xyzs = np.array(
                [
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1],
                ],
                dtype=float,
            )
            xyzs[:, 0] *= dimensions[0] / 2
            xyzs[:, 1] *= dimensions[1] / 2
            xyzs[:, 2] *= dimensions[2] / 2
            self.B_r_CM = B_r_CP + xyzs @ A_BM.T
            self.cells = [(VTK_LAGRANGE_HEXAHEDRON, range(8))]

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                vtk_points = r_OC + self.B_r_CM @ A_IB.T
                return vtk_points, self.cells, {}, {}

    return _Box


def Cone(Base):
    class _Cone(Base):
        def __init__(
            self,
            radius=1,
            height=2,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            # compute inertia quantities of body
            if density is not None:
                mass = density / 3 * height * np.pi * radius**2
                B_Theta_C = (
                    np.diag(
                        [
                            0.15 * radius**2 + 0.1 * height**2,
                            0.15 * radius**2 + 0.1 * height**2,
                            0.3 * radius**2,
                        ]
                    )
                    * mass
                )

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            super().__init__(**kwargs)
            # mesh for visualization
            phis = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
            xyz1 = np.stack(
                [
                    np.cos(phis) * radius,
                    np.sin(phis) * radius,
                    np.zeros_like(phis),
                ],
                axis=1,
            )
            xyz2 = np.stack(
                [
                    np.zeros_like(phis),
                    np.zeros_like(phis),
                    np.ones_like(phis) * height,
                ],
                axis=1,
            )
            phis2 = phis + (np.pi / 3.0)
            xyz3 = np.stack(
                [
                    np.cos(phis2) * 2.0 * radius,
                    np.sin(phis2) * 2.0 * radius,
                    np.zeros_like(phis2),
                ],
                axis=1,
            )
            xyz4 = np.stack(
                [
                    np.zeros_like(phis),
                    np.zeros_like(phis),
                    np.ones_like(phis2) * height,
                ],
                axis=1,
            )
            self.B_r_CM = (
                B_r_CP + np.concatenate([xyz1, xyz2, xyz3, xyz4], axis=0) @ A_BM.T
            )
            self.cells = [(VTK_BEZIER_WEDGE, range(12))]
            self.point_data = {
                "RationalWeights": np.vstack([1] * 6 + [0.5] * 6),
            }
            self.cell_data = {
                "HigherOrderDegrees": [[2, 2, 1]],
            }

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                vtk_points = r_OC + self.B_r_CM @ A_IB.T
                return vtk_points, self.cells, self.point_data, self.cell_data

    return _Cone


def Cylinder(Base):
    class _Cylinder(Base):
        def __init__(
            self,
            radius=1,
            height=2,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            # compute inertia quantities of body
            if density is not None:
                mass = density * height * np.pi * radius**2
                B_Theta_C = (
                    np.diag(
                        [
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.5 * radius**2,
                        ]
                    )
                    * mass
                )

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            super().__init__(**kwargs)
            # mesh for visualization
            phis = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
            xyz1 = np.stack(
                [
                    np.cos(phis) * radius,
                    np.sin(phis) * radius,
                    np.ones_like(phis) * (-height / 2),
                ],
                axis=1,
            )
            xyz2 = np.stack(
                [
                    np.cos(phis) * radius,
                    np.sin(phis) * radius,
                    np.ones_like(phis) * (height / 2),
                ],
                axis=1,
            )
            phis2 = phis + (np.pi / 3.0)
            xyz3 = np.stack(
                [
                    np.cos(phis2) * 2.0 * radius,
                    np.sin(phis2) * 2.0 * radius,
                    np.ones_like(phis2) * (-height / 2),
                ],
                axis=1,
            )
            xyz4 = np.stack(
                [
                    np.cos(phis2) * 2.0 * radius,
                    np.sin(phis2) * 2.0 * radius,
                    np.ones_like(phis2) * (height / 2),
                ],
                axis=1,
            )
            self.B_r_CM = (
                B_r_CP + np.concatenate([xyz1, xyz2, xyz3, xyz4], axis=0) @ A_BM.T
            )
            self.cells = [(VTK_BEZIER_WEDGE, range(12))]
            self.point_data = {
                "RationalWeights": np.vstack([1] * 6 + [0.5] * 6),
            }
            self.cell_data = {
                "HigherOrderDegrees": [[2, 2, 1]],
            }

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                vtk_points = r_OC + self.B_r_CM @ A_IB.T
                return vtk_points, self.cells, self.point_data, self.cell_data

    return _Cylinder


def Sphere(Base):
    class _Sphere(Base):
        def __init__(
            self,
            radius=1,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            **kwargs,
        ):
            self.radius = radius
            # compute inertia quantities of body
            if density is not None:
                mass = density * 4 / 3 * np.pi * radius**3
                B_Theta_C = np.eye(3) * 2 / 5 * mass * radius**2

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            super().__init__(**kwargs)
            # mesh for visualization
            # Farin, Piper and Worsey (1987)
            # The octant of a sphere as a non-degenerate triangular Bézier patch
            c1 = (np.sqrt(3) - 1.0) / np.sqrt(3)
            c2 = (np.sqrt(3) + 1.0) / (2.0 * np.sqrt(3))
            c3 = 1.0 - (5.0 - np.sqrt(2)) * (7.0 - np.sqrt(3)) / 46.0
            xyzs = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, c1, 0],
                        [c2, c2, 0],
                        [c1, 1, 0],
                        [0, 1, c1],
                        [0, c2, c2],
                        [0, c1, 1],
                        [c1, 0, 1],
                        [c2, 0, c2],
                        [1, 0, c1],
                        [1, c3, c3],
                        [c3, 1, c3],
                        [c3, c3, 1],
                    ]
                )
                * radius
            )
            w1 = 4 * np.sqrt(3) * (np.sqrt(3) - 1.0)
            w2 = 3 * np.sqrt(2)
            w3 = np.sqrt(2.0 / 3.0) * (3.0 + 2.0 * np.sqrt(2) - np.sqrt(3))
            weights = np.array(
                [
                    w1,
                    w1,
                    w1,
                    w2,
                    4.0,
                    w2,
                    w2,
                    4.0,
                    w2,
                    w2,
                    4.0,
                    w2,
                    w3,
                    w3,
                    w3,
                ]
            ).reshape(-1, 1)
            pids = [np.arange(15)]
            # flip octant to get full sphere
            for ax in range(3):
                flip_mask = np.array([1, 1, 1])
                flip_mask[ax] = -1
                npts = len(xyzs)
                sel = xyzs[:, np.where(flip_mask == -1)[0][0]] != 0
                ids_old = np.argwhere(sel).flatten()
                ids_new = np.arange(np.sum(sel)) + npts
                xyzs = np.append(xyzs, xyzs[sel] * flip_mask, axis=0)
                weights = np.append(weights, weights[sel], axis=0)
                for i in range(len(pids)):
                    ids = pids[i].copy()
                    sort = np.array(
                        [np.where(ids_old == x)[0][0] for x in ids if x in ids_old]
                    )
                    ids[np.isin(ids, ids_old)] = ids_new[sort]
                    pids.append(ids)
            self.B_r_CM = B_r_CP + xyzs @ A_BM.T
            self.cells = [(VTK_BEZIER_TRIANGLE, ids) for ids in pids]
            self.point_data = {
                "RationalWeights": weights,
            }
            self.cell_data = {
                "HigherOrderDegrees": [[4, 4, 0]] * 8,
            }

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                vtk_points = r_OC + self.B_r_CM @ A_IB.T
                return vtk_points, self.cells, self.point_data, self.cell_data

    return _Sphere


def Capsule(Base):
    class _Capsule(Base):
        def __init__(
            self,
            radius=1,
            height=2,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            # compute inertia quantities of body
            if density is not None:
                # https://www.gamedev.net/tutorials/programming/math-and-physics/capsule-inertia-tensor-r3856/
                m_cyl = density * height * np.pi * radius**2
                m_cap = density * 2 / 3 * np.pi * radius**3
                mass = m_cyl + 2 * m_cap
                B_Theta_C = (
                    np.diag(
                        [
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.5 * radius**2,
                        ]
                    )
                    * m_cyl
                    + np.diag(
                        [
                            0.4 * radius**2 + 0.5 * height**2 + 3 / 8 * height * radius,
                            0.4 * radius**2 + 0.5 * height**2 + 3 / 8 * height * radius,
                            0.4 * radius**2,
                        ]
                    )
                    * 2
                    * m_cap
                )

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            super().__init__(**kwargs)
            # mesh for caps, similar to sphere
            c1 = (np.sqrt(3) - 1.0) / np.sqrt(3)
            c2 = (np.sqrt(3) + 1.0) / (2.0 * np.sqrt(3))
            c3 = 1.0 - (5.0 - np.sqrt(2)) * (7.0 - np.sqrt(3)) / 46.0
            xyzs = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, c1, 0],
                        [c2, c2, 0],
                        [c1, 1, 0],
                        [0, 1, c1],
                        [0, c2, c2],
                        [0, c1, 1],
                        [c1, 0, 1],
                        [c2, 0, c2],
                        [1, 0, c1],
                        [1, c3, c3],
                        [c3, 1, c3],
                        [c3, c3, 1],
                    ]
                )
                * radius
            )
            w1 = 4 * np.sqrt(3) * (np.sqrt(3) - 1.0)
            w2 = 3 * np.sqrt(2)
            w3 = np.sqrt(2.0 / 3.0) * (3.0 + 2.0 * np.sqrt(2) - np.sqrt(3))
            weights_sph = np.array(
                [
                    w1,
                    w1,
                    w1,
                    w2,
                    4.0,
                    w2,
                    w2,
                    4.0,
                    w2,
                    w2,
                    4.0,
                    w2,
                    w3,
                    w3,
                    w3,
                ]
            ).reshape(-1, 1)
            pids = [np.arange(15)]
            # flip octant to get full sphere
            for ax in range(3):
                if ax == 2:
                    xyzs[:, 2] += height / 2
                flip_mask = np.array([1, 1, 1])
                flip_mask[ax] = -1
                npts = len(xyzs)
                sel = xyzs[:, np.where(flip_mask == -1)[0][0]] != 0
                ids_old = np.argwhere(sel).flatten()
                ids_new = np.arange(np.sum(sel)) + npts
                xyzs = np.append(xyzs, xyzs[sel] * flip_mask, axis=0)
                weights_sph = np.append(weights_sph, weights_sph[sel], axis=0)
                for i in range(len(pids)):
                    ids = pids[i].copy()
                    sort = np.array(
                        [np.where(ids_old == x)[0][0] for x in ids if x in ids_old]
                    )
                    ids[np.isin(ids, ids_old)] = ids_new[sort]
                    pids.append(ids)
            B_r_CM_sph = B_r_CP + xyzs @ A_BM.T
            cells_sph = [(VTK_BEZIER_TRIANGLE, ids) for ids in pids]

            # mesh for caps
            c1 = (np.sqrt(3) - 1.0) / np.sqrt(3)
            c2 = (np.sqrt(3) + 1.0) / (2.0 * np.sqrt(3))
            xyzs = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, c1, 0],
                        [c2, c2, 0],
                        [c1, 1, 0],
                    ]
                )
                * radius
            )
            diff = np.array([0, 0, height / 2])
            xyzs = np.vstack((xyzs - diff, xyzs + diff))
            w1 = 4 * np.sqrt(3) * (np.sqrt(3) - 1.0)
            w2 = 3 * np.sqrt(2)
            weights_cyl = np.array(
                [
                    w1,
                    w1,
                    w2,
                    4.0,
                    w2,
                ]
            ).reshape(-1, 1)
            weights_cyl = np.vstack((weights_cyl, weights_cyl))
            pids = [np.array([0, 1, 6, 5, 2, 3, 4, 7, 8, 9])]
            # flip octant to get full cylinder
            for ax in range(2):
                flip_mask = np.array([1, 1, 1])
                flip_mask[ax] = -1
                npts = len(xyzs)
                sel = xyzs[:, np.where(flip_mask == -1)[0][0]] != 0
                ids_old = np.argwhere(sel).flatten()
                ids_new = np.arange(np.sum(sel)) + npts
                xyzs = np.append(xyzs, xyzs[sel] * flip_mask, axis=0)
                weights_cyl = np.append(weights_cyl, weights_cyl[sel], axis=0)
                for i in range(len(pids)):
                    ids = pids[i].copy()
                    sort = np.array(
                        [np.where(ids_old == x)[0][0] for x in ids if x in ids_old]
                    )
                    ids[np.isin(ids, ids_old)] = ids_new[sort]
                    pids.append(ids)
            B_r_CM_cyl = B_r_CP + xyzs @ A_BM.T
            cells_cyl = [
                (VTK_BEZIER_QUADRILATERAL, ids + len(B_r_CM_sph)) for ids in pids
            ]

            self.B_r_CM = np.vstack((B_r_CM_sph, B_r_CM_cyl))
            self.cells = cells_sph + cells_cyl
            self.point_data = {
                "RationalWeights": np.vstack((weights_sph, weights_cyl)),
            }
            self.cell_data = {
                "HigherOrderDegrees": [[4, 4, 0]] * 8 + [[4, 1, 0]] * 4,
            }
            print()

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                vtk_points = r_OC + self.B_r_CM @ A_IB.T
                return vtk_points, self.cells, self.point_data, self.cell_data

    return _Capsule


def Tetrahedron(Base):
    class _Tetrahedron(Base):
        def __init__(
            self,
            edge=1,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            **kwargs,
        ):
            # compute inertia quantities of body
            if density is not None:
                mass = density * edge**3 * np.sqrt(2) / 12
                B_Theta_C = np.eye(3) * mass * edge**2 / 20

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            super().__init__(**kwargs)
            # mesh for visualization
            # see https://de.wikipedia.org/wiki/Tetraeder
            h_D = edge * np.sqrt(3) / 2
            h_P = edge * np.sqrt(2 / 3)
            r_OC = np.array([0, h_D / 3, h_P / 4])
            p1 = np.array([-edge / 2, 0, 0]) - r_OC
            p2 = np.array([+edge / 2, 0, 0]) - r_OC
            p3 = np.array([0, h_D, 0]) - r_OC
            p4 = np.array([0, h_D / 3, h_P]) - r_OC
            vertices = np.vstack((p1, p2, p3, p4))
            self.B_r_CM = B_r_CP + vertices @ A_BM.T
            self.cells = [(VTK_LAGRANGE_TETRAHEDRON, range(4))]

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                vtk_points = r_OC + self.B_r_CM @ A_IB.T
                return vtk_points, self.cells, {}, {}

    return _Tetrahedron


def Axis(Base):
    MeshedBase = Meshed(Base)

    class _Axis(MeshedBase):
        def __init__(
            self,
            origin_size=0.04,
            **kwargs,
        ):
            trimesh_obj = trimesh.creation.axis(origin_size=origin_size)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Axis
