"""Ring of Lorenz-96 sites around a latitude circle, advected by their neighbours, damped, and forced."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, vector

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class Lorenz96Element(Element):
    """Ring of Lorenz-96 sites around a latitude circle, advected by their neighbours, damped, and forced.

    Every site follows `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`, with the indices taken around the ring.

    Args:
        label:
            The label of the ring.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial values of all sites.

        F:
            The constant forcing.

    Examples:
        >>> from skcomplex.models.earth.climate import Lorenz96Element
        >>> Lorenz96Element('ring').F
        8.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 8.0,
        F: Annotated[float, ParameterSpec()] = 8.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.F = F

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the advection, damping, and forcing of every site of the ring.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the values of all sites.
        """
        x = s.x
        n_sites = len(x)
        rates = [(x[(i + 1) % n_sites] - x[i - 2]) * x[i - 1] - x[i] + s.F for i in range(n_sites)]
        return [RateLaw(s.x, vector(SR, rates))]
