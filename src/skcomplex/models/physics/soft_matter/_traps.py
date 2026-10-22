"""Trapping: walkers driven through quenched landscapes of broadly distributed trapping times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, state
from ....core.model import Element, LatticeSpace, Law, LocalRule, Neighborhood, Observable, Symbols, System
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]

ALONG_ROW = Neighborhood(((0, 0), (0, -1), (0, 1)))
VISITED = VisualSpec('image', {'states': ['visited'], 'low': 0.0, 'high': 1.0})


def _hop(*values: Array) -> tuple[Array, Array, Array]:
    """Return the occupation, the visits, and the clocks after one step of fully biased hops out of the traps."""
    occupied, left_occupied, _, visited, _, _, clock, left_clock, _, tau, left_tau, right_tau = values
    leaves = (occupied == 1) & (clock + 1 >= tau) & (right_tau > 0)
    arrives = (left_occupied == 1) & (left_clock + 1 >= left_tau)
    stays = (occupied == 1) & ~leaves
    next_occupied = (stays | arrives).astype(float)
    return next_occupied, np.maximum(visited, next_occupied), np.where(stays, clock + 1, 0.0)


def _mean_position(s: Symbols) -> object:
    """Return the mean distance travelled by the walkers."""
    return s.visits / s.walkers - 1


class TrapSite(Element):
    """Site of a line of traps, which holds a walker for its trapping time `tau` before the walker hops to the right.

    The walker on a site counts the steps it has spent there with `clock` and hops to the next site after `tau`
    steps, and every site keeps a record of whether a walker has visited it. A walker hops only onto a site of positive
    trapping time, so on a lattice that reads `-1` outside its edges a walker on the last site stays there.

    Args:
        label:
            The label of the site or the lattice group.

        occupied:
            Whether a walker is on the site.

        visited:
            Whether a walker has been on the site.

        clock:
            The number of steps the walker has spent on the site.

        tau:
            The trapping time of the site.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import TrapSite
        >>> TrapSite('site', tau=3.0).tau
        3.0
    """

    occupied: float = state(default=0.0, domain=Domain.BOOLEAN)
    visited: float = state(default=0.0, domain=Domain.BOOLEAN)
    clock: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    tau: float = state(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the deterministic hops of the walkers out of their traps.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        targets = (s.occupied, s.visited, s.clock)
        return [LocalRule(targets, ALONG_ROW, _hop, reads=(*targets, s.tau))]


def _trap_observables(system: System) -> list[Observable]:
    """Return the visits, the walkers, and the mean position of the traps."""
    return [
        Observable('visits', group='traps', variable='visited', reduce='sum'),
        Observable('walkers', group='traps', variable='occupied', reduce='sum'),
        Observable('position', _mean_position),
    ]


@register_model(
    field='physics/soft_matter',
    tags=('lattice', 'anomalous diffusion', 'disorder', 'traps', 'subdiffusion'),
    references=('Bertin 2021', 'Bouchaud and Georges 1990'),
    defaults={'until': 2000, 'dt': 10},
    observables=_trap_observables,
)
def biased_trap_walk(
    n_walkers: int = 100,
    *,
    n_sites: int = 300,
    alpha: float = 0.5,
    tau0: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return walkers that hop always to the right through quenched traps with broadly distributed trapping times.

    Every walker moves along its own line of sites and stays `tau_i` steps on site `i` before it hops to site
    `i + 1`. The trapping times are drawn once, as `ceil(tau0 u^(-1 / alpha))` for uniform numbers `u`, so their
    distribution decays as `tau0^alpha / tau^(1 + alpha)`. For `alpha < 1` their mean diverges, the time to cross `N`
    sites grows as `N^(1 / alpha)`, and the displacement grows subdiffusively as `x ~ (t / tau0)^alpha`, with the
    mean `<x> = (t / tau0)^alpha sin(pi alpha) / (pi alpha)` at long times. The walkers start on the first site.

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

    Returns:
        system:
            The model, with the lattice `traps` of shape `(n_walkers, n_sites)` and the mean displacement `position`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import biased_trap_walk
        >>> biased_trap_walk(4, n_sites=10).elements[0].size
        40
    """
    shape = (n_walkers, n_sites)
    tau = np.ceil(tau0 * (1 - np.random.default_rng(random_state).random(shape)) ** (-1 / alpha))
    start = np.zeros(shape)
    start[:, 0] = 1.0
    traps = TrapSite.lattice(
        'traps',
        LatticeSpace(shape, boundary='fixed', fill=-1.0),
        visual=VISITED,
        occupied=start,
        visited=start,
        tau=tau,
    )
    return System(components=[traps])
