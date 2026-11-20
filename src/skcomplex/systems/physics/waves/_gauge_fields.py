"""Lattice gauge fields.

The module covers angle links sampled by the Wilson action, and lattice Maxwell fields from static charges.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg

from ....core.components import Observable
from ....core.components.elements import Element, Period, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import codifferential, exterior_derivative
from ....core.components.spaces import BaseSpace, CellComplexSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.runtime import BaseClock, BaseObserver, Trajectory
from ....core.engine.solvers import BaseSolver, MetropolisSolver, VelocityVerletSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.fields import WilsonSamplingInteraction

DYNAMICS = ('monte-carlo', 'hamiltonian')
MAXWELL_SOLVER = VelocityVerletSolver(dt=0.02)
MAXWELL_UNTIL = 50.0
MAXWELL_DT = 0.5
SOLVER_TOLERANCE = 1e-15
NEUTRAL_TOTAL = 1e-12


class GaugeLinkElement(Element):
    """Link of a lattice with an angle `theta`.

    The angle is the U(1) or Z_N variable of a lattice gauge theory.

    Args:
        label:
            The label of the link or the group on the links.

        theta:
            The angle of the link, or one angle per link, which wraps around the circle.

        space:
            The cell complex of the links, or `None` for one link.

        on:
            `'edges'` for one link on every edge of the complex, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.waves import GaugeLinkElement
        >>> GaugeLinkElement('link', theta=0.5).theta
        0.5
    """

    visual = VisualSpec('cell', {'state': 'theta'})

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.theta = theta


class MaxwellLinkElement(Element):
    """Link of a lattice with a vector potential `A`.

    Its velocity `A'` is the field `-E`.

    Args:
        label:
            The label of the link or the group on the links.

        A:
            The initial vector potential, or one per link.

        space:
            The cell complex of the links, or `None` for one link.

        on:
            `'edges'` for one link on every edge of the complex, or `None`.

        values:
            Further values by name, such as the initial velocity `A'`, minus the electric field, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.waves import MaxwellLinkElement
        >>> MaxwellLinkElement('link', A=0.5).A
        0.5
    """

    visual = VisualSpec('cell', {'state': 'A'})

    def __init__(
        self,
        label: str,
        *,
        A: object = 0.0,
        space: BaseSpace | None = None,
        on: str | None = None,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values, space=space, on=on)
        self.A = A


class StaticChargeElement(Element):
    """Site of a lattice with a static electric charge `q`.

    Args:
        label:
            The label of the site or the group on the sites.

        q:
            The electric charge, or one charge per site.

        space:
            The cell complex of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.waves import StaticChargeElement
        >>> StaticChargeElement('site', q=-1.0).q
        -1.0
    """

    def __init__(
        self,
        label: str,
        *,
        q: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.q = q


class MaxwellInteraction(Interaction):
    """Lattice Maxwell equations of a potential `A` on the links of a cell complex.

    The potential follows `A'' = -codifferential(dA)`, so its velocity `A' = -E` and the field `E` follows
    `E' = codifferential(dA)`, as light in a vacuum or a wave on a crystal lattice.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the links with the potential `A`.

    Examples:
        >>> from skcomplex.systems.physics.waves import MaxwellInteraction
        >>> MaxwellInteraction('maxwell', target='links').target
        'links'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the potential.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            potential = context.target.A
            self.laws_ = [RateLaw(potential.diff(context.t, 2), -codifferential(exterior_derivative(potential)))]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        electric = -np.ravel(np.asarray(values["links.A'"], dtype=float))
        charges = np.ravel(np.asarray(values['charges.q'], dtype=float))
        return float(np.abs(-(edges.T @ electric) - charges).max())

    @staticmethod
    def _get_field_energy(faces: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
        """Return the energy of the electric and magnetic fields.

        The electric field lives on the links and the magnetic field on the plaquettes.
        """
        del t
        velocity = np.ravel(np.asarray(values["links.A'"], dtype=float))
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.dimension = dimension
        self.n = n
        self.beta = beta
        self.dynamics = dynamics
        self.charges = charges
        self.angles = angles
        self.amplitude = amplitude
        self.seed = seed

    def _get_maxwell_system(self, space: CellComplexSpace) -> System:
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
        place = cast(LatticeSpace, space.space)
        sources = space.edges_[None][0]
        along_last = space.link_axes_ == place.n_dim_ - 1
        first = np.unravel_index(sources, place.shape_)[0]
        wave = self.amplitude * np.cos(2 * np.pi * first / place.shape_[0]) * along_last
        links = MaxwellLinkElement(
            'links',
            A=wave,
            space=space,
            on='edges',
            values={"A'": -LatticeGaugeFieldsSystem._get_electric_field(space, density)},
        )
        sites = StaticChargeElement('charges', q=density, space=space)
        maxwell = MaxwellInteraction('maxwell', target='links')
        return System(self.label, elements=[links, sites], interactions=[maxwell], solver=MAXWELL_SOLVER)

    def _get_monte_carlo_system(self, space: CellComplexSpace) -> System:
        """Return U(1) or Z_N link angles sampled by the Wilson action under discrete time."""
        rng = np.random.default_rng(self.seed)
        size = space.incidences_[0].shape[0]
        values = () if self.n is None else tuple(2 * np.pi * np.arange(self.n) / self.n)
        drawn = rng.uniform(-np.pi, np.pi, size) if self.n is None else rng.choice(values, size)
        links = GaugeLinkElement(
            'links',
            theta=drawn if self.angles is None else np.asarray(self.angles, dtype=float),
            space=space,
            on='edges',
        )
        sampling = WilsonSamplingInteraction('wilson', target='links', beta=self.beta)
        if self.n is None:
            solver = MetropolisSolver(moves={'links.theta': 'rotate'}, schedule='colored', width=1.5)
        else:
            solver = MetropolisSolver(
                moves={'links.theta': 'random-value'},
                rule='heat-bath',
                schedule='colored',
                values={'links.theta': values},
            )
        return System(self.label, elements=[links], interactions=[sampling], solver=solver)

    def _get_gauge_observables(self, space: CellComplexSpace) -> list[Observable]:
        """Return the observables of the dynamics.

        They are the mean plaquette of a sampled system, or the Gauss residual and the field energy of a Maxwell one.
        """
        if self.dynamics != 'hamiltonian':
            return [
                Observable(
                    'mean_plaquette',
                    function=partial(LatticeGaugeFieldsSystem._get_mean_plaquette, space.incidences_[1]),
                ),
            ]
        return [
            Observable(
                'gauss_residual',
                function=partial(LatticeGaugeFieldsSystem._get_gauss_residual, space.incidences_[0]),
            ),
            Observable(
                'field_energy',
                function=partial(LatticeGaugeFieldsSystem._get_field_energy, space.incidences_[1]),
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the link variables on the cell complex of a periodic lattice, sampled or evolved in real time.

        Args:
            context:
                The observables to compile with the problem, or `None` for those of the dynamics.

        Raises:
            ValidationError:
                The dynamics are unknown, or the charges do not hold one value per site of total 0.
        """
        super()._prepare(context)
        if self.dynamics not in DYNAMICS:
            error_msg = f'Unknown dynamics {self.dynamics!r}. Use one of {", ".join(DYNAMICS)}.'
            raise ValidationError(error_msg)
        space = CellComplexSpace(LatticeSpace((self.size,) * self.dimension)).prepare()
        if not context:
            self.observables_ = tuple(self._get_gauge_observables(space))
        if self.dynamics == 'hamiltonian':
            self.system_ = self._get_maxwell_system(space)
        else:
            self.system_ = self._get_monte_carlo_system(space)

    def simulate(
        self,
        until: float | None = None,
        *,
        dt: float | None = None,
        initial_state: Mapping[str, object] | None = None,
        random_state: int | np.random.Generator | None = None,
        observables: Sequence[Observable] = (),
        observers: Sequence[BaseObserver] = (),
        invariants: Mapping[str, float] | None = None,
        schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None = None,
        clock: BaseClock | None = None,
        checked: bool = False,
        record_draws: bool = False,
        replay: Trajectory | None = None,
    ) -> Trajectory:
        """Simulate the prepared system, for 50 units of time every 0.5 with Hamiltonian dynamics by default.

        The arguments are those of `BaseSystem.simulate`, whose end time and time between frames default to those of
        the dynamics.

        Returns:
            trajectory:
                The states and observables of the run at every frame.
        """
        if self.dynamics == 'hamiltonian':
            until = MAXWELL_UNTIL if until is None else until
            dt = MAXWELL_DT if dt is None else dt
        return super().simulate(
            until,
            dt=dt,
            initial_state=initial_state,
            random_state=random_state,
            observables=observables,
            observers=observers,
            invariants=invariants,
            schedule=schedule,
            clock=clock,
            checked=checked,
            record_draws=record_draws,
            replay=replay,
        )
