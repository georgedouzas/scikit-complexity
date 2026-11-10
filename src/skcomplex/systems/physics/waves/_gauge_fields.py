"""Lattice gauge fields.

The module covers angle links sampled by the Wilson action, and lattice Maxwell fields from static charges.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sage.all__sagemath_symbolics import Expression, pi

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import codifferential, exterior_derivative
from ....core.components.spaces import CellComplexSpace, DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.engine.solvers import VelocityVerletSolver
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.fields import WilsonSamplingInteraction

DYNAMICS = ('monte-carlo', 'hamiltonian')
SOLVER_TOLERANCE = 1e-15
NEUTRAL_TOTAL = 1e-12


class GaugeLinkElement(Element):
    """Link of a lattice with an angle `theta`.

    The angle is the U(1) or Z_N variable of a lattice gauge theory.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        theta:
            The angle of the link.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import GaugeLinkElement
        >>> GaugeLinkElement('link', theta=0.5).theta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta


class MaxwellLinkElement(Element):
    """Link of a lattice with a vector potential `A` of second order.

    Its velocity `A_d1` is the field `-E`.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        A:
            The initial vector potential.

        A_d1:
            The initial velocity of the vector potential, minus the electric field.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import MaxwellLinkElement
        >>> MaxwellLinkElement('link', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(order=2)] = 0.0,
        A_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.A_d1 = A_d1


class StaticChargeElement(Element):
    """Site of a lattice with a static electric charge `q`.

    Args:
        label:
            The label of the site or the group on the sites.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        q:
            The electric charge.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import StaticChargeElement
        >>> StaticChargeElement('site', q=-1.0).q
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        q: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.q = q


class LatticeGaugeFieldsSystem(BaseSystem):
    """Link variables on the cell complex of a periodic lattice.

    The variables are sampled or evolved in real time.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The dynamics are unknown, or the charges do not hold one value per site of total 0.

    Examples:
        >>> from skcomplex.systems.physics.waves import LatticeGaugeFieldsSystem
        >>> [element.label for element in LatticeGaugeFieldsSystem(4).prepare().system_.elements]
        ['links']
        >>> system = LatticeGaugeFieldsSystem(4, dynamics='hamiltonian').prepare().system_
        >>> [element.label for element in system.elements]
        ['links', 'charges']
    """

    field = 'physics/waves'
    topics = ('gauge', 'lattice', 'monte carlo', 'maxwell', 'cell complex')
    references = ('Schwichtenberg 2020', 'Wilson 1974')
    until = 5000
    dt = 1

    @staticmethod
    def _get_mean_plaquette(faces: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
        """Return the mean cosine of the oriented link angle sums around the plaquettes."""
        del t
        return float(np.cos(faces @ np.ravel(np.asarray(values['links.theta'], dtype=float))).mean())

    @staticmethod
    def _get_gauss_residual(edges: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
        """Return the Gauss residual of the electric field.

        The residual is the largest absolute difference of the codifferential of the field and the site charge.
        """
        del t
        electric = -np.ravel(np.asarray(values['links.A_d1'], dtype=float))
        charges = np.ravel(np.asarray(values['charges.q'], dtype=float))
        return float(np.abs(-(edges.T @ electric) - charges).max())

    @staticmethod
    def _get_field_energy(faces: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
        """Return the energy of the electric and magnetic fields.

        The electric field lives on the links and the magnetic field on the plaquettes.
        """
        del t
        velocity = np.ravel(np.asarray(values['links.A_d1'], dtype=float))
        magnetic = faces @ np.ravel(np.asarray(values['links.A'], dtype=float))
        return float((velocity @ velocity + magnetic @ magnetic) / 2)

    @staticmethod
    def _get_electric_field(space: CellComplexSpace, charges: Array) -> Array:
        """Return the electric field of least norm whose codifferential is the charge.

        The codifferential matches the charge of every site.
        """
        field = scipy.sparse.linalg.lsqr(
            -space.incidences_[0].T,
            charges,
            atol=SOLVER_TOLERANCE,
            btol=SOLVER_TOLERANCE,
            conlim=0.0,
            iter_lim=100 * charges.size,
        )[0]
        return np.asarray(field, dtype=float)

    @staticmethod
    def _get_curl_of_curl(namespace: SimpleNamespace) -> Expression:
        """Return the rate of the velocity of the potential.

        The rate is minus the codifferential of its exterior derivative.
        """
        return -codifferential(exterior_derivative(namespace.target.A))

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.dimension = dimension
        self.n = n
        self.beta = beta
        self.dynamics = dynamics
        self.charges = charges
        self.angles = angles
        self.amplitude = amplitude
        self.seed = seed

    def _maxwell_system(self, space: CellComplexSpace) -> CompositeSystem:
        """Return a lattice Maxwell field on the links.

        The field comes from static charges and a transverse wave.
        """
        density = np.zeros(space.size_) if self.charges is None else np.asarray(self.charges, dtype=float)
        if density.size != space.size_ or abs(density.sum()) > NEUTRAL_TOTAL:
            error_msg = (
                f'The charges are {density.size} values of total {density.sum()}, but a periodic lattice takes one '
                f'charge per site of its {space.size_} sites, of total 0.'
            )
            raise ValidationError(error_msg)
        place = space.space
        sources = space.edges_[None][0]
        along_last = space.link_axes_ == place.n_dim_ - 1
        first = np.unravel_index(sources, place.shape_)[0]
        wave = self.amplitude * np.cos(2 * np.pi * first / place.shape_[0]) * along_last
        links = GroupElement(
            'links',
            element=MaxwellLinkElement,
            space=space,
            on='edges',
            values={'A': wave, 'A_d1': -LatticeGaugeFieldsSystem._get_electric_field(space, density)},
            visual=VisualSpec('cell', {'state': 'A'}),
        )
        sites = GroupElement('charges', element=StaticChargeElement, space=space, on='sites', values={'q': density})
        rate = RateInteraction('maxwell', target='links', state='A', rate=LatticeGaugeFieldsSystem._get_curl_of_curl)
        return CompositeSystem(self.label, elements=[links, sites], interactions=[rate])

    def _monte_carlo_system(self, space: CellComplexSpace) -> CompositeSystem:
        """Return U(1) or Z_N link angles sampled by the Wilson action under discrete time."""
        rng = np.random.default_rng(self.seed)
        size = space.incidences_[0].shape[0]
        values = () if self.n is None else tuple(2 * np.pi * np.arange(self.n) / self.n)
        drawn = rng.uniform(-np.pi, np.pi, size) if self.n is None else rng.choice(values, size)
        links = GroupElement(
            'links',
            element=GaugeLinkElement,
            space=space,
            on='edges',
            values={'theta': drawn if self.angles is None else np.asarray(self.angles, dtype=float)},
            visual=VisualSpec('cell', {'state': 'theta'}),
        )
        move = {'move': 'rotate'} if self.n is None else {'move': 'random-value', 'values': values, 'rule': 'heat-bath'}
        sampling = WilsonSamplingInteraction('wilson', target='links', beta=self.beta, **move)
        return CompositeSystem(self.label, elements=[links], interactions=[sampling], time=DiscreteTime())

    @staticmethod
    def _gauge_observables(system: CompositeSystem) -> list[ObservableSpec]:
        """Return the observables of the dynamics.

        They are the mean plaquette of a sampled system, or the Gauss residual and the field energy of a Maxwell one.

        Args:
            system:
                The parts of the system.

        Returns:
            observables:
                The observables.
        """
        links = next(element for element in system.elements if element.label == 'links')
        space = links.space.prepare()
        if links.element is GaugeLinkElement:
            return [
                ObservableSpec(
                    'mean_plaquette',
                    function=partial(LatticeGaugeFieldsSystem._get_mean_plaquette, space.incidences_[1]),
                ),
            ]
        return [
            ObservableSpec(
                'gauss_residual',
                function=partial(LatticeGaugeFieldsSystem._get_gauss_residual, space.incidences_[0]),
            ),
            ObservableSpec(
                'field_energy',
                function=partial(LatticeGaugeFieldsSystem._get_field_energy, space.incidences_[1]),
            ),
        ]

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The settings have the observables of the dynamics, and velocity Verlet for Maxwell.
        """
        settings = super().get_run_settings()
        settings['observables'] = tuple(self._gauge_observables(self._build_system()))
        if self.dynamics == 'hamiltonian':
            settings['until'] = 50.0
            settings['dt'] = 0.5
            settings['solver'] = VelocityVerletSolver(dt=0.02)
        return settings

    def _build_system(self) -> CompositeSystem:
        """Return link variables on the cell complex of a periodic lattice.

        The variables are sampled or evolved in real time.
        """
        if self.dynamics not in DYNAMICS:
            error_msg = f'Unknown dynamics {self.dynamics!r}. Use one of {", ".join(DYNAMICS)}.'
            raise ValidationError(error_msg)
        space = CellComplexSpace(LatticeSpace((self.size,) * self.dimension)).prepare()
        if self.dynamics == 'hamiltonian':
            return self._maxwell_system(space)
        return self._monte_carlo_system(space)
