"""Invasion fronts of a population that grows logistically and diffuses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from ....core.components.spaces import LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.length = length
        self.D = D
        self.r = r
        self.spacing = spacing
        self.front = front

    def _get_element_label(self) -> str:
        """Return the label of the field of the sites."""
        return 'line'

    def _get_space(self) -> LatticeSpace:
        """Return the line of the sites, whose reflecting ends let no density through."""
        return LatticeSpace(self.length, spacing=self.spacing, boundary='reflecting')

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        u = (np.arange(self.length) < self.front * self.length).astype(float)
        return {
            'u': u,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'D': self.D,
            'r': self.r,
        }

    def _get_equations(self) -> Equations:
        """Return the logistic growth of the density."""
        return {'u': lambda s: s.target.r * s.target.u * (1 - s.target.u)}

    def _get_diffusivities(self) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the diffusion coefficient of the density."""
        return {'u': lambda s: s.target.D}
