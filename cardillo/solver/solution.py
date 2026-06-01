from collections import namedtuple
import dill


def save_solution(sol, filename):
    """Store a `Solution` object into a given file."""
    with open(filename, mode="wb") as f:
        dill.dump(sol, f)


def load_solution(filename):
    """Load a `Solution` object from a given file."""
    with open(filename, mode="rb") as f:
        return dill.load(f)


class Solution:
    """Class to store solver outputs."""

    def __init__(
        self,
        system,
        t,
        q,
        u=None,
        u_dot=None,
        la_g=None,
        la_c=None,
        **kwargs,
    ):
        self.system = system
        self.t = t
        self.q = q
        self.u = u
        self.u_dot = u_dot
        self.la_g = la_g
        self.la_c = la_c

        self.__dict__.update(**kwargs)

    def save(self, filename):
        save_solution(self, filename)

    def _get_value(self, key, idx):
        r = self.__dict__[key]
        if r is None:
            return None
        else:
            return r[idx]

    def __iter__(self):
        keys = [k for k in self.__dict__ if k not in ["system"]]
        Result = namedtuple("Result", keys)

        for idx in range(len(self.t)):
            yield Result(*(self._get_value(k, idx) for k in keys))
