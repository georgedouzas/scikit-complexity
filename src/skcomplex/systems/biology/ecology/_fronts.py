"""Invasion fronts of a population that grows logistically and diffuses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.spaces import LatticeSpace
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import ReactionDiffusionSystem
from ...general.dynamics.types import Equations


class FisherKPPSystem(ReactionDiffusionSystem):
    """The Fisher-KPP equation, whose step invades the empty line as a front.

    The front has the speed `2 sqrt(r D)`.

    The system has the density `line.u`.

    Args:
        length:
            The number of sites of the line.

        D:
            The diffusion coefficient.

        r:
            The growth rate.

        spacing:
            The distance between sites.

        front:
            The fraction of the line that starts occupied, on its left.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import FisherKPPSystem
        >>> int(FisherKPPSystem(100, front=0.1).prepare().system_.elements[0].values['u'].sum())
        10
    """

    field = 'biology/ecology'
    topics = ('field', 'fronts', 'reaction-diffusion', 'pde')
    references = ('Fisher 1937', 'Kolmogorov, Petrovsky, and Piskunov 1937')
    until = 40.0
    dt = 1.0

    def __init__(
        self,
        length: int = 800,
        *,
        D: float = 1.0,
        r: float = 1.0,
        spacing: float = 0.25,
        front: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.length = length
        self.D = D
        self.r = r
        self.spacing = spacing
        self.front = front

    def _get_element_label(self) -> str:
        """Return the label of the field of the sites."""
        return 'line'

    def _get_space(self) -> LatticeSpace:
        """Return the line of the sites."""
        return LatticeSpace(self.length, spacing=self.spacing)

    def _get_boundaries(self) -> dict[str, str]:
        """Return the zero-flux boundary of the density."""
        return {'u': 'zero-flux'}

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the density, occupied on the left of the line, and the parameters."""
        u = (np.arange(self.length) < self.front * self.length).astype(float)
        return {
            'u': DeclaredVariableSpec(u, StateSpec(domain=RealSet.closed_open(0, oo))),
            'D': DeclaredVariableSpec(self.D, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'r': DeclaredVariableSpec(self.r, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the logistic growth of the density."""
        return {'u': lambda s: s.target.r * s.target.u * (1 - s.target.u)}

    def _get_diffusivities(self) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the diffusion coefficient of the density."""
        return {'u': lambda s: s.target.D}
