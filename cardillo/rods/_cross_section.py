import numpy as np

class CircularCrossSection:
    def __init__(self, radius):
        """Circular cross-section.

        Parameters
        ----------
        radius : float
            Radius of the cross-section
        """
        self._variable = callable(radius)
        self._radius = lambda xi: radius(xi) if self._variable else radius
        self._area = (
            lambda xi: np.pi * radius(xi) ** 2
            if self._variable
            else np.pi * radius**2
        )
        # see https://en.wikipedia.org/wiki/First_moment_of_area
        self._first_moment = (lambda xi: np.zeros(3)) if self._variable else np.zeros(3)
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self._second_moment = (
            lambda xi: np.diag([2, 1, 1]) / 4 * np.pi * radius(xi) ** 4
            if self._variable
            else np.diag([2, 1, 1]) / 4 * np.pi * radius**4
        )

    @property
    def area(self):
        return self._area

    @property
    def first_moment(self):
        return self._first_moment

    @property
    def second_moment(self):
        return self._second_moment

    @property
    def radius(self):
        return self._radius
