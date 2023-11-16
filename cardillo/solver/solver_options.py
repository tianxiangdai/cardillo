from dataclasses import dataclass

from numpy import max, abs


@dataclass
class SolverOptions:
    fixed_point_atol: float = 1e-8
    fixed_point_rtol: float = 1e-8
    fixed_point_max_iter: int = int(1e3)
    newton_max_iter: int = 20
    newton_atol: float = 1e-8
    newton_rtol: float = 1e-8
    error_function: callable = lambda x: max(abs(x))
    prox_scaling: float = 1.0
    continue_with_unconverged: bool = True
