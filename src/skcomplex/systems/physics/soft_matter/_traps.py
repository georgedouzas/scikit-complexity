"""Trapping.

Walkers are driven through quenched landscapes of broadly distributed trapping times.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array

ALONG_ROW = LatticeNeighborhood(((0, 0), (0, -1), (0, 1)))
VISITED = VisualSpec('image', {'states': ['visited'], 'low': 0.0, 'high': 1.0})


class TrapSiteElement(Element):
    """Site of a line of traps.

    The site holds a walker for its trapping time `tau` before the walker hops to the right.

    The walker on a site counts the steps it has spent there with `clock`, and every site keeps a record of whether a
    walker has visited it.

    Args:
        label:
            The label of the site or the lattice of sites.

        occupied:
            Whether a walker is on the site.

        visited:
            Whether a walker has been on the site.

        clock:
            The number of steps the walker has spent on the site.

        tau:
            The trapping time of the site.

        space:
            The lattice whose sites hold one site each, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TrapSiteElement
        >>> TrapSiteElement('site', tau=3.0).tau
        3.0
    """

    visual = VISITED

    def __init__(
        self,
        label: str,
        *,
        occupied: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        visited: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        clock: Annotated[object, NN] = 0.0,
        tau: Annotated[object, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.occupied = occupied
        self.visited = visited
        self.clock = clock
        self.tau = tau


class BiasedHopInteraction(Interaction):
    """Hops of walkers that always move to the right out of their traps.

    A walker on a site hops to the next site after `tau` steps there, and only onto a site of positive trapping time,
    so on a lattice that reads `-1` outside its edges a walker on the last site stays there. It is a carrier driven
    through a disordered wire, or a motor protein stalling on a damaged track.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the lattice of sites.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import BiasedHopInteraction
        >>> BiasedHopInteraction('hops', target='traps').target
        'traps'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _hop(*values: Array) -> tuple[Array, Array, Array]:
        """Return the occupation, the visits, and the clocks after one step.

        The step is one of fully biased hops out of the traps.
        """
        occupied, left_occupied, _, visited, _, _, clock, left_clock, _, tau, left_tau, right_tau = values
        leaves = (occupied == 1) & (clock + 1 >= tau) & (right_tau > 0)
        arrives = (left_occupied == 1) & (left_clock + 1 >= left_tau)
        stays = (occupied == 1) & ~leaves
        next_occupied = (stays | arrives).astype(float)
        return next_occupied, np.maximum(visited, next_occupied), np.where(stays, clock + 1, 0.0)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the hops of the walkers on the sites of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            states = (s.occupied, s.visited, s.clock)
            self.laws_ = [
                UpdateLaw(states, self._hop, neighborhood=ALONG_ROW, reads=(*states, s.tau), label=self.label),
            ]


def _get_mean_position(s: SimpleNamespace) -> object:
    """Return the mean distance travelled by the walkers."""
    return s.visits / s.walkers - 1


class BiasedTrapWalkSystem(BaseSystem):
    """Walkers that hop always to the right through quenched traps.

    The trapping times are broadly distributed.

    Every walker moves along its own line of sites and stays `tau_i` steps on site `i` before it hops to site `i + 1`.
    The trapping times are drawn once, as `ceil(tau0 u^(-1 / alpha))` for uniform numbers `u`, so their distribution
    decays as `tau0^alpha / tau^(1 + alpha)`. For `alpha < 1` their mean diverges, the time to cross `N` sites grows as
    `N^(1 / alpha)`, and the displacement grows subdiffusively as `x ~ (t / tau0)^alpha`, with the mean `<x> = (t /
    tau0)^alpha sin(pi alpha) / (pi alpha)` at long times. The walkers start on the first site. The system has the
    lattice `traps` of shape `(n_walkers, n_sites)` and the mean displacement `position`.

    Args:
        n_walkers:
            The number of walkers, each in its own landscape.

        n_sites:
            The number of sites of every landscape.

        alpha:
            The exponent of the distribution of trapping times.

        tau0:
            The smallest trapping time.

        random_state:
            The seed or generator of the trapping times.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import BiasedTrapWalkSystem
        >>> BiasedTrapWalkSystem(4, n_sites=10).prepare().system_.elements[0].space.prepare().size_
        40
    """

    field = 'physics/soft_matter'
    topics = ('lattice', 'anomalous diffusion', 'disorder', 'traps', 'subdiffusion')
    references = ('Bertin 2021', 'Bouchaud and Georges 1990')
    until = 2000
    dt = 10
    observables = (
        Observable('visits', group='traps', variable='visited', reduce='sum'),
        Observable('walkers', group='traps', variable='occupied', reduce='sum'),
        Observable('position', _get_mean_position),
    )

    def __init__(
        self,
        n_walkers: int = 100,
        *,
        n_sites: int = 300,
        alpha: float = 0.5,
        tau0: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.n_sites = n_sites
        self.alpha = alpha
        self.tau0 = tau0
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the lines of traps with the walkers on their first sites.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = (self.n_walkers, self.n_sites)
        tau = np.ceil(self.tau0 * (1 - np.random.default_rng(self.random_state).random(shape)) ** (-1 / self.alpha))
        start = np.zeros(shape)
        start[:, 0] = 1.0
        traps = TrapSiteElement(
            'traps',
            occupied=start,
            visited=start,
            clock=np.zeros(shape),
            tau=tau,
            space=LatticeSpace(shape, boundary='fixed', fill=-1.0),
        )
        self.system_ = System(self.label, elements=[traps], interactions=[BiasedHopInteraction('hops', target='traps')])
