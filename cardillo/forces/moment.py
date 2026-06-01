from numpy import einsum, zeros
from vtk import VTK_VERTEX


class B_Moment:
    r"""Moment represented w.r.t. body-fixed B-basis

    Parameters
    ----------
    moment : np.ndarray (3,)
        Moment w.r.t. body-fixed B-basis as a callable function of time t.
    subsystem : object
        Object on which moment acts.
    xi : #TODO
    name : str
        Name of contribution.
    """

    def __init__(self, moment, subsystem, xi=zeros(3), name="moment"):
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return self.moment(t) @ self.B_J_R(t, q)

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.moment(t), self.B_J_R_q(t, q))
