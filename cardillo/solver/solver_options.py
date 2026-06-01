from dataclasses import dataclass
from scipy.sparse.linalg import spsolve


@dataclass
class SolverOptions:
    newton_atol: float = 1e-10
    newton_rtol: float = 1e-6
    newton_max_iter: int = 20
    continue_with_unconverged: bool = False
    linear_solver: callable = spsolve
    compute_consistent_initial_conditions: bool = True

    def __post_init__(self):
        assert self.newton_atol > 0
        assert self.newton_rtol > 0
        assert self.newton_max_iter > 0
