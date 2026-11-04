"""Lattice gauge fields: angle links sampled by the Wilson action, and lattice Maxwell fields from static charges."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sage.all__sagemath_symbolics import Expression

from ....core.elements import GroupElement
from ....core.interactions import RateInteraction
from ....core.laws.utils import codifferential, exterior_derivative
from ....core.spaces import CellComplexSpace, DiscreteTime, LatticeSpace
from ....core.system import System
from ....engine.solvers import VelocityVerletSolver
from ....errors import ValidationError
from ....specs import ObservableSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._gauge_link import GaugeLinkElement
from ._maxwell_link import MaxwellLinkElement
from ._static_charge import StaticChargeElement
from ._wilson_sampling import WilsonSamplingInteraction

DYNAMICS = ('monte-carlo', 'hamiltonian')
SOLVER_TOLERANCE = 1e-15
NEUTRAL_TOTAL = 1e-12


def _mean_plaquette(faces: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
    """Return the mean cosine of the oriented sums of the link angles around the plaquettes."""
    del t
    return float(np.cos(faces @ np.ravel(np.asarray(values['links.theta'], dtype=float))).mean())


def _gauss_residual(edges: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
    """Return the largest absolute difference of the codifferential of the electric field and the site charge."""
    del t
    electric = -np.ravel(np.asarray(values['links.A_d1'], dtype=float))
    charges = np.ravel(np.asarray(values['charges.q'], dtype=float))
    return float(np.abs(-(edges.T @ electric) - charges).max())


def _field_energy(faces: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
    """Return the energy of the electric field on the links and of the magnetic field on the plaquettes."""
    del t
    velocity = np.ravel(np.asarray(values['links.A_d1'], dtype=float))
    magnetic = faces @ np.ravel(np.asarray(values['links.A'], dtype=float))
    return float((velocity @ velocity + magnetic @ magnetic) / 2)


def _gauge_observables(system: System) -> list[ObservableSpec]:
    """Return the mean plaquette of a sampled system, or the Gauss residual and the field energy of a Maxwell one."""
    links = next(element for element in system.elements if element.label == 'links')
    space = links.space.check().prepare()
    if links.element is GaugeLinkElement:
        return [ObservableSpec('mean_plaquette', function=partial(_mean_plaquette, space.incidences_[1]))]
    return [
        ObservableSpec('gauss_residual', function=partial(_gauss_residual, space.incidences_[0])),
        ObservableSpec('field_energy', function=partial(_field_energy, space.incidences_[1])),
    ]


def _electric_field(space: CellComplexSpace, charges: Array) -> Array:
    """Return the electric field of the smallest norm whose codifferential is the charge of every site."""
    field = scipy.sparse.linalg.lsqr(
        -space.incidences_[0].T,
        charges,
        atol=SOLVER_TOLERANCE,
        btol=SOLVER_TOLERANCE,
        conlim=0.0,
        iter_lim=100 * charges.size,
    )[0]
    return np.asarray(field, dtype=float)


def _curl_of_curl(namespace: SimpleNamespace) -> Expression:
    """Return the rate of the velocity of the potential, minus the codifferential of its exterior derivative."""
    return -codifferential(exterior_derivative(namespace.target.A))


def _maxwell_system(space: CellComplexSpace, charges: Sequence[float] | None, amplitude: float) -> System:
    """Return a lattice Maxwell field on the links from static charges and a transverse wave."""
    density = np.zeros(space.size_) if charges is None else np.asarray(charges, dtype=float)
    if density.size != space.size_ or abs(density.sum()) > NEUTRAL_TOTAL:
        error_msg = (
            f'The charges are {density.size} values of total {density.sum()}, but a periodic lattice takes one charge '
            f'per site of its {space.size_} sites, of total 0.'
        )
        raise ValidationError(error_msg)
    place = space.space
    sources = space.edges_[None][0]
    along_last = space.link_axes_ == place.n_dim_ - 1
    first = np.unravel_index(sources, place.shape_)[0]
    wave = amplitude * np.cos(2 * np.pi * first / place.shape_[0]) * along_last
    links = GroupElement(
        'links',
        element=MaxwellLinkElement,
        space=space,
        on='edges',
        values={'A': wave, 'A_d1': -_electric_field(space, density)},
        visual=VisualSpec('cell', {'state': 'A'}),
    )
    sites = GroupElement('charges', element=StaticChargeElement, space=space, on='sites', values={'q': density})
    rate = RateInteraction('maxwell', target='links', state='A', rate=_curl_of_curl)
    return System(elements=[links, sites], interactions=[rate])


def _monte_carlo_system(
    space: CellComplexSpace,
    n: int | None,
    beta: float,
    angles: Sequence[float] | None,
    seed: int,
) -> System:
    """Return U(1) or Z_N link angles sampled by the Wilson action under discrete time."""
    rng = np.random.default_rng(seed)
    size = space.incidences_[0].shape[0]
    values = () if n is None else tuple(2 * np.pi * np.arange(n) / n)
    drawn = rng.uniform(-np.pi, np.pi, size) if n is None else rng.choice(values, size)
    links = GroupElement(
        'links',
        element=GaugeLinkElement,
        space=space,
        on='edges',
        values={'theta': drawn if angles is None else np.asarray(angles, dtype=float)},
        visual=VisualSpec('cell', {'state': 'theta'}),
    )
    move = {'move': 'rotate'} if n is None else {'move': 'random-value', 'values': values, 'rule': 'heat-bath'}
    sampling = WilsonSamplingInteraction('wilson', target='links', beta=beta, **move)
    return System(elements=[links], interactions=[sampling], time=DiscreteTime())


def _gauge_settings(dynamics: str = 'monte-carlo', **_parameters: object) -> dict[str, object]:
    """Return the run settings of the Maxwell dynamics, and none for Monte Carlo, the catalog defaults."""
    return {'until': 50.0, 'dt': 0.5, 'solver': VelocityVerletSolver(dt=0.02)} if dynamics == 'hamiltonian' else {}


@register_model(
    field='physics/waves',
    tags=('gauge', 'lattice', 'monte carlo', 'maxwell', 'cell complex'),
    references=('Schwichtenberg 2020', 'Wilson 1974'),
    defaults={'until': 5000, 'dt': 1},
    settings=_gauge_settings,
    observables=_gauge_observables,
)
def lattice_gauge_fields(
    size: int = 32,
    *,
    dimension: int = 2,
    n: int | None = None,
    beta: float = 2.0,
    dynamics: str = 'monte-carlo',
    charges: Sequence[float] | None = None,
    angles: Sequence[float] | None = None,
    amplitude: float = 0.1,
    seed: int = 0,
) -> System:
    """Return link variables on the cell complex of a periodic lattice, sampled or evolved in real time.

    With Monte Carlo dynamics, link angles sample the Wilson action `-beta sum_p cos(theta_p)` under discrete time, as
    U(1) angles or as the angles `2 pi k / n` of Z_n. With Hamiltonian dynamics, a potential on the links follows the
    lattice Maxwell equations `A_t = -E` and `E_t = codifferential(dA)` under continuous time, from static charges on
    the sites and a transverse wave. In two dimensions the U(1) mean plaquette at large size is `I_1(beta) / I_0(beta)`.

    Args:
        size:
            The number of sites along every axis.

        dimension:
            The number of axes of the lattice.

        n:
            The order of the Z_n group, or `None` for U(1).

        beta:
            The inverse coupling of the Wilson action.

        dynamics:
            `'monte-carlo'` or `'hamiltonian'`.

        charges:
            The static charge of every site, of total 0, or `None` for none, with Hamiltonian dynamics.

        angles:
            The initial link angles, or `None` for random angles, with Monte Carlo dynamics.

        amplitude:
            The amplitude of the initial transverse wave, with Hamiltonian dynamics.

        seed:
            The seed of the initial link angles.

    Returns:
        system:
            The system.

    Raises:
        ValidationError:
            The dynamics are unknown, or the charges do not hold one value per site of total 0.

    Examples:
        >>> from skcomplex.models.physics.waves import lattice_gauge_fields
        >>> [element.label for element in lattice_gauge_fields(4).elements]
        ['links']
        >>> [element.label for element in lattice_gauge_fields(4, dynamics='hamiltonian').elements]
        ['links', 'charges']
    """
    if dynamics not in DYNAMICS:
        error_msg = f'Unknown dynamics {dynamics!r}. Use one of {", ".join(DYNAMICS)}.'
        raise ValidationError(error_msg)
    space = CellComplexSpace(LatticeSpace((size,) * dimension)).check().prepare()
    if dynamics == 'hamiltonian':
        return _maxwell_system(space, charges, amplitude)
    return _monte_carlo_system(space, n, beta, angles, seed)
