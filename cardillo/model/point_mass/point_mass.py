import numpy as np

class Point_mass():
    def __init__(self, m, dim=3, q0=None, u0=None):
        self.m = m
        self.nq = dim
        self.nu = dim

        self.M_ = m * np.eye(dim)

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, M_coo):
        M_coo.extend(self.M_, (self.uDOF, self.uDOF))

    def B_dense(self, t, q):
        return np.eye(self.nq)

    def B(self, t, q, B_coo):
        B_coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def point(self, ID=None):
        return Point(self, ID)

class Point():
    def __init__(self, subsystem, ID):
        self.subsystem = subsystem
        self.qDOF = subsystem.qDOF
        self.nq = subsystem.nq
        self.ID = ID

        self.B = subsystem.B_dense

    def position(self, t, q):
        return q.copy().resize((3,))

    def position_q(self, t, q):
        return np.eye(3, self.nq)

if __name__ == "__main__":
    m = 1
    from cardillo.model import Model
    from cardillo.model.point_mass import Point_mass

    model = Model()
    PM = Point_mass(2, dim=2)
    model.add(PM)
    model.assemble()

    print(f'M = \n{model.M(0, model.q0).toarray()}')
    print(f'B = \n{model.B(0, model.q0).toarray()}')