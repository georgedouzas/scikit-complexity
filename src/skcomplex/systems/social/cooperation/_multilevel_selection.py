"""Multilevel selection of altruistic punishment.

It follows the densities of group compositions under group-level competition.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, tanh
from sage.modules.free_module_element import FreeModuleElement
from scipy.stats import irwinhall

from ....core.components import Observable
from ....core.components.elements import VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import coordinate, divergence, integral_transform, integration_variable
from ....core.components.spaces import AxisSpec, FiniteVolumeSpec, LatticeSpace, PhaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.kinetics import ActivityDensityElement

PAIRWISE_VICTORIES = ('fermi', 'local', 'tullock')
TRIMORPHIC_VICTORIES = ('defectors', 'payoff')
ADDITIVE_VICTORIES = ('punishers', 'payoff')
SIMPLEX_SAMPLES = 401
REFERENCES = ('Cooney 2024',)
DISCRETIZED_OPERATORS = ('laplacian', 'partial', 'upwind', 'inverse_helmholtz')


type Fraction = Expression | npt.NDArray[np.float64]


class SimplexSpace(PhaseSpace):
    """A space whose extra axes are frequencies that sum to at most one.

    A simplex is the set of frequencies that are nonnegative and sum to one, such as the shares of three strategies. The
    class axes run over `[0, 1]` with the same number of cells, and the last frequency is one minus the sum of the
    others, so two class axes hold the simplex of three frequencies. The cells cut by the slanted face are partial, and
    each cell and face keeps the fraction of its volume or area inside the simplex. Finite-volume fluxes pass only
    through the open part of each face, and reductions over the class axes weight each cell by its volume.

    Unlike `PhaseSpace`, whose class axes are independent, it constrains the frequencies to a simplex. Choose it when a
    density lives over the shares of several types.

    It appears in many fields. Populations evolve over the shares of strategies in evolutionary game theory, genotypes
    over allele frequencies in population genetics, and voters over the shares of parties.

    Args:
        position:
            The lattice of the positions.

        axes:
            The class axes, each on `[0, 1]` with the same number of cells and not periodic.

        limit:
            The largest number of sites of the space.

        label:
            The name of the space in its system, such as `'grid'`.

    Attributes:
        volumes_:
            The fraction of the volume of every cell inside the simplex, with one entry along every position axis.

        apertures_:
            The fraction of the area inside the simplex of every face along every axis of the product, one more face
            than cells along a class axis, and one entry along the position axes.

    Raises:
        PhaseSpaceTooLargeError:
            The product has more sites than the limit.

    Examples:
        >>> from skcomplex.core.components.spaces import AxisSpec, LatticeSpace
        >>> from skcomplex.systems.social.cooperation import SimplexSpace
        >>> axes = [AxisSpec('x', 10, 0.0, 1.0), AxisSpec('y', 10, 0.0, 1.0)]
        >>> space = SimplexSpace(LatticeSpace((1,)), axes).prepare()
        >>> space.shape_, float(space.volumes_.sum())
        ((1, 10, 10), 50.0)
        >>> sorted(set(space.volumes_.ravel().tolist()))
        [0.0, 0.5, 1.0]
    """

    def _prepare(self, context: None = None) -> None:
        """Set the sites of the phase space and the cell volumes and face apertures.

        It first raises an error for class axes that are not intervals on `[0, 1]` of equal cells. They must have the
        same number of cells.
        """
        super()._prepare(context)
        sizes = {int(axis.size) for axis in self.axes}
        unit = all(axis.low == 0.0 and axis.high == 1.0 and not axis.periodic for axis in self.axes)
        if len(sizes) != 1 or not unit:
            error_msg = (
                'The class axes of a simplex space are intervals on [0, 1] with the same number of cells, such as '
                f"`AxisSpec('x', 30, 0.0, 1.0)`, but they are {list(self.axes)!r}."
            )
            raise ValidationError(error_msg)
        dimensions = len(self.axes)
        cells = int(self.axes[0].size)
        positions = (1,) * len(self.position_axes_)
        indices = np.indices((cells,) * dimensions).sum(axis=0)
        self.volumes_ = np.reshape(self._get_fraction(cells - indices, dimensions), (*positions, *indices.shape))
        apertures = [np.ones((1,) * self.n_dim_) for _ in self.position_axes_]
        for axis in range(dimensions):
            faces = [cells] * dimensions
            faces[axis] = cells + 1
            grid = np.indices(faces)
            others = grid.sum(axis=0) - grid[axis]
            fraction = self._get_fraction(cells - grid[axis] - others, dimensions - 1)
            apertures.append(np.reshape(fraction, (*positions, *fraction.shape)))
        self.apertures_ = tuple(apertures)

    def _get_volumes(self) -> npt.NDArray[np.float64]:
        """Return the fraction of the volume of every cell inside the simplex."""
        return self.volumes_

    def _get_apertures(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the fraction of every face inside the simplex along every axis."""
        return self.apertures_

    def _supports_operator(self, name: str) -> bool:
        """Check whether a field on the simplex takes an operator.

        The partial cells along the slanted face take finite-volume fluxes, reductions, and integrals, but no local
        operator of finite differences, such as `laplacian`, nor a spectral inversion.
        """
        return name not in DISCRETIZED_OPERATORS

    @staticmethod
    def _get_fraction(room: npt.NDArray[np.int_], dimensions: int) -> npt.NDArray[np.float64]:
        """Return the fraction of a unit cube whose coordinates sum to at most the room.

        The cube has a dimension, and the room is that of every cell.
        """
        if dimensions == 0:
            return (room > 0).astype(float)
        return np.asarray(irwinhall(dimensions).cdf(room), dtype=float)


class GroupDensityElement(ActivityDensityElement):
    """Density of groups over their strategy fractions, drawn as a simplex."""

    visual = VisualSpec('image', {'slices': {0: 0}})


class SelectionDriftInteraction(Interaction):
    """Selection drift `du/dt + divergence(a u) = source(ns)` of a density `u` over a trait.

    Evolutionary game theory names it the selection within groups, the drift of a density of traits by selection.
    The drift field is `a(ns)`. The flux of the conservation law is the drift times the state, written in conservative
    form, so the total of the state on a periodic field stays constant and, on a spectral field, the dealiasing of
    products applies to it. The drift is one expression for a field of one dimension, or one per dimension, of the
    states and parameters of the field, and may read other states than the one it carries, such as the flow of a fluid
    derived from a streamfunction, or the drift of a probability density in a Fokker-Planck equation. Its laws are
    `RateLaw(u, -divergence(flux))` and, with a source, `RateLaw(u, source)`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the field whose state is carried.

        state:
            The name of the carried state of the field.

        drift:
            A function of the namespace that returns the drift, or one drift per dimension.

        source:
            A function of the namespace that returns the source, or `None` for none.

    Raises:
        ValidationError:
            The state is not a name, or a function is not callable.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.social.cooperation import SelectionDriftInteraction
        >>> space = LatticeSpace((50,), spacing=0.1, boundary='periodic', discretization='finite-volume')
        >>> values = {'u': np.exp(-((np.arange(50) - 25) / 5.0) ** 2)}
        >>> drift = SelectionDriftInteraction('drift', target='tracer', state='u', drift=lambda s: 1.0)
        >>> system = System(elements=[Element('tracer', values=values, space=space)], interactions=[drift])
        >>> trajectory = system.prepare().simulate(until=1.0, dt=0.01)
        >>> total = trajectory.get_values('tracer.u')[-1].sum()
        >>> round(float(total), 6) == round(float(values['u'].sum()), 6)
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'u',
        drift: Callable[[SimpleNamespace], object] | None = None,
        source: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.drift = drift
        self.source = source

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The state is not a name, or the drift or the source is not callable.
        """
        super()._check_arguments()
        if not isinstance(self.state, str):
            error_msg = f'The state of {self.label!r} is {self.state!r}. Give the name of a state of the field.'
            raise ValidationError(error_msg)
        if not callable(self.drift) or (self.source is not None and not callable(self.source)):
            error_msg = f'The drift and the source of {self.label!r} are functions of the namespace.'
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of the divergence of the flux and of the source, after checking the functions.

        Args:
            context:
                The namespace of the field.
        """
        super()._prepare(context)
        state = getattr(context.target, self.state)
        flux = self._get_flux(context)
        components = list(flux) if isinstance(flux, list) else [flux]
        self.laws_ = [RateLaw(state, -divergence(components))]
        if self.source is not None:
            self.laws_.append(RateLaw(state, self.source(context)))

    def _get_flux(self, namespace: SimpleNamespace) -> object:
        """Return the flux `a u`, one component per component of the drift.

        Args:
            namespace:
                The namespace of the symbols the flux can use, with the namespace of the field.

        Returns:
            flux:
                The flux, one expression or one per dimension.
        """
        state = getattr(namespace.target, self.state)
        drift = self.drift(namespace) if callable(self.drift) else None
        if isinstance(drift, FreeModuleElement | list | tuple):
            return [SR(component) * state for component in drift]
        return SR(drift) * state


class NonlinearPairwiseGroupConflictSystem(BaseSystem):
    """The density of groups of defectors and altruistic punishers.

    The groups are under pairwise group conflict. The density `f(z)` of groups with a fraction `z` of punishers is
    advected by the replicator dynamics within groups, `z (1 - z) (pi_P - pi_D)`, and groups win pairwise conflicts,
    copying themselves over the loser, at the rate `group_strength f(z) int (rho(z, u) - rho(u, z)) f(u) du`. The
    victory probability `rho` is not additively separable: the group Fermi rule `(1 + tanh(s (G(z) - G(u)))) / 2`
    (`'fermi'`), the local rule normalized by the payoffs of both groups (`'local'`), or the Tullock contest function of
    the payoffs above their minimum with the exponent `1 / a` (`'tullock'`), where `G(z)` is the average payoff of a
    group (Cooney 2024, arXiv 2405.18419 §3.1 and §6). The density starts uniform, and the integral against the
    two-point kernel `rho(z, u) - rho(u, z)` keeps its total at one.

    The system has the density `groups.x`.

    Args:
        cells:
            The number of cells of the axis of the fraction of punishers.

        victory:
            The victory probability, `'fermi'`, `'local'`, or `'tullock'`.

        group_strength:
            The relative strength `lambda` of between-group conflict.

        benefit:
            The benefit `b` of the donation game.

        cost:
            The cost `c` of cooperating.

        punishment:
            The punishment `p` of a defector by every punisher it meets.

        interaction_cost:
            The cost `k` a punisher pays per defector it meets.

        fixed_cost:
            The fixed cost `q` of being able to punish.

        sensitivity:
            The sensitivity `s` of the Fermi rule.

        exponent:
            The exponent `a` of the Tullock contest function.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import NonlinearPairwiseGroupConflictSystem
        >>> NonlinearPairwiseGroupConflictSystem(40).prepare().system_.elements[0].x.size
        40
    """

    field = 'social/cooperation'
    topics = ('field', 'multilevel selection', 'punishment', 'integral operator', 'pde')
    references = REFERENCES
    until = 15.0
    dt = 0.5

    @staticmethod
    def _get_pairwise_payoff(z: Fraction, payoffs: tuple[float, ...]) -> Fraction:
        """Return the average payoff `G(z)` of a group of punishers `z` and defectors.

        It follows equation 2.11.
        """
        benefit, cost, punishment, interaction, fixed = payoffs
        return (benefit - cost - fixed - punishment - interaction) * z + (punishment + interaction) * z**2

    @staticmethod
    def _check_victory(victory: str, victories: tuple[str, ...]) -> None:
        """Raise an error for a victory probability that is not one of the given ones."""
        if victory not in victories:
            error_msg = f'Unknown victory probability {victory!r}. Use one of {", ".join(victories)}.'
            raise ValidationError(error_msg)

    def __init__(
        self,
        cells: int = 100,
        *,
        victory: str = 'fermi',
        group_strength: float = 2.0,
        benefit: float = 2.0,
        cost: float = 1.0,
        punishment: float = 0.5,
        interaction_cost: float = 0.1,
        fixed_cost: float = 0.0,
        sensitivity: float = 2.0,
        exponent: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.cells = cells
        self.victory = victory
        self.group_strength = group_strength
        self.benefit = benefit
        self.cost = cost
        self.punishment = punishment
        self.interaction_cost = interaction_cost
        self.fixed_cost = fixed_cost
        self.sensitivity = sensitivity
        self.exponent = exponent

    @staticmethod
    def _get_pairwise_victory(
        first: Expression,
        second: Expression,
        payoffs: tuple[float, ...],
        *,
        victory: str,
        sensitivity: float,
        exponent: float,
    ) -> Expression:
        """Return the probability that a group of punishers defeats another.

        The groups are `first` and `second`. It follows equations 3.12 to 3.14.
        """
        own, other = (
            NonlinearPairwiseGroupConflictSystem._get_pairwise_payoff(item, payoffs) for item in (first, second)
        )
        if victory == 'fermi':
            return (1 + tanh(sensitivity * (own - other))) / 2
        if victory == 'local':
            return (1 + (own - other) / (abs(own) + abs(other))) / 2
        centres = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
        lowest = float(
            np.min(
                np.asarray(NonlinearPairwiseGroupConflictSystem._get_pairwise_payoff(centres, payoffs), dtype=float),
            ),
        )
        own_share, other_share = (own - lowest) ** (1 / exponent), (other - lowest) ** (1 / exponent)
        return own_share / (own_share + other_share)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        NonlinearPairwiseGroupConflictSystem._check_victory(self.victory, PAIRWISE_VICTORIES)
        payoffs = (self.benefit, self.cost, self.punishment, self.interaction_cost, self.fixed_cost)
        z, u = coordinate('z'), integration_variable('z')
        win = self._get_pairwise_victory(
            z,
            u,
            payoffs,
            victory=self.victory,
            sensitivity=self.sensitivity,
            exponent=self.exponent,
        )
        loss = self._get_pairwise_victory(
            u,
            z,
            payoffs,
            victory=self.victory,
            sensitivity=self.sensitivity,
            exponent=self.exponent,
        )
        gain = win - loss
        advantage = (self.punishment + self.interaction_cost) * z - (
            self.cost + self.fixed_cost + self.interaction_cost
        )
        position = LatticeSpace((1,), discretization='finite-volume', scheme=FiniteVolumeSpec(limiter=None))
        space = PhaseSpace(position, [AxisSpec('z', self.cells, 0.0, 1.0)])
        groups = ActivityDensityElement('groups', x=np.ones((1, self.cells)), space=space)
        within = SelectionDriftInteraction(
            'within',
            target='groups',
            state='x',
            drift=lambda _: [0, z * (1 - z) * advantage],
        )
        between = Interaction(
            'between',
            roles={'target': 'groups'},
            laws=lambda s: [
                RateLaw(
                    s.target.x.diff(s.t),
                    self.group_strength * s.target.x * integral_transform(s.target.x, gain, ('z',)),
                ),
            ],
        )
        self.system_ = System(self.label, elements=[groups], interactions=[within, between], solver=RK4Solver(dt=0.01))


class TwoLevelReplicatorSystem(BaseSystem):
    """The two-level replicator equation of defectors and altruistic punishers.

    The group conflict is additive and pairwise. The density `f(z)` of groups with a fraction `z` of punishers is
    advected by the replicator dynamics within groups, `z (1 - z) (pi_P - pi_D)`, and groups replicate at the rate
    `group_strength f(z) (G(z) - int G(u) f(u) du)` (Cooney 2024, arXiv 2405.18419 Eq. 3.3). Pairwise group conflict
    with an additively separable victory probability `rho(z, u) = 1/2 + (G(z) - G(u)) / 2` reduces to this equation (Eq.
    3.11): `G(z) = z` for the probability of Boyd and coauthors based on the fraction of punishers (`'punishers'`, Eq.
    3.6 and section 5.1), or the average payoff of a group normalized by its range, `G(z) / (G* - G_*)` (`'payoff'`, Eq.
    3.8 and section 5.2). From a uniform density with `G(z) = z` and `p < c + q`, the groups concentrate at the
    all-defector state below the threshold `lambda* = c + q - p`, and above it the mean fraction of punishers tends to
    `1 - (c + q - p) / lambda` (Eq. 5.4).

    The system has the density `groups.x`.

    Args:
        cells:
            The number of cells of the axis of the fraction of punishers.

        victory:
            The group-level victory probability, `'punishers'` or `'payoff'`.

        group_strength:
            The relative strength `lambda` of between-group competition.

        benefit:
            The benefit `b` of the donation game.

        cost:
            The cost `c` of cooperating.

        punishment:
            The punishment `p` of a defector by every punisher it meets.

        interaction_cost:
            The cost `k` a punisher pays per defector it meets.

        fixed_cost:
            The fixed cost `q` of being able to punish.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import TwoLevelReplicatorSystem
        >>> TwoLevelReplicatorSystem(40).prepare().system_.elements[0].x.size
        40
    """

    field = 'social/cooperation'
    topics = ('field', 'multilevel selection', 'punishment', 'two-level replicator', 'pde')
    references = REFERENCES
    until = 40.0
    dt = 1.0

    def __init__(
        self,
        cells: int = 100,
        *,
        victory: str = 'punishers',
        group_strength: float = 2.0,
        benefit: float = 2.0,
        cost: float = 1.0,
        punishment: float = 0.5,
        interaction_cost: float = 0.5,
        fixed_cost: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.cells = cells
        self.victory = victory
        self.group_strength = group_strength
        self.benefit = benefit
        self.cost = cost
        self.punishment = punishment
        self.interaction_cost = interaction_cost
        self.fixed_cost = fixed_cost

    @staticmethod
    def _get_group_rate(z: Expression, payoffs: tuple[float, ...], victory: str) -> Expression:
        """Return the net group-level replication rate `G(z)`.

        The victory probability is additively separable.
        """
        if victory == 'punishers':
            return z
        centres = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
        values = np.asarray(NonlinearPairwiseGroupConflictSystem._get_pairwise_payoff(centres, payoffs), dtype=float)
        return NonlinearPairwiseGroupConflictSystem._get_pairwise_payoff(z, payoffs) / float(
            values.max() - values.min(),
        )

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        NonlinearPairwiseGroupConflictSystem._check_victory(self.victory, ADDITIVE_VICTORIES)
        payoffs = (self.benefit, self.cost, self.punishment, self.interaction_cost, self.fixed_cost)
        z, u = coordinate('z'), integration_variable('z')
        gain = self._get_group_rate(z, payoffs, self.victory) - self._get_group_rate(u, payoffs, self.victory)
        advantage = (self.punishment + self.interaction_cost) * z - (
            self.cost + self.fixed_cost + self.interaction_cost
        )
        position = LatticeSpace((1,), discretization='finite-volume', scheme=FiniteVolumeSpec(limiter=None))
        space = PhaseSpace(position, [AxisSpec('z', self.cells, 0.0, 1.0)])
        groups = ActivityDensityElement('groups', x=np.ones((1, self.cells)), space=space)
        within = SelectionDriftInteraction(
            'within',
            target='groups',
            state='x',
            drift=lambda _: [0, z * (1 - z) * advantage],
        )
        between = Interaction(
            'between',
            roles={'target': 'groups'},
            laws=lambda s: [
                RateLaw(
                    s.target.x.diff(s.t),
                    self.group_strength * s.target.x * integral_transform(s.target.x, gain, ('z',)),
                ),
            ],
        )
        self.system_ = System(self.label, elements=[groups], interactions=[within, between], solver=RK4Solver(dt=0.01))


class TrimorphicMultilevelSelectionSystem(BaseSystem):
    """The density of groups of cooperators, defectors, and altruistic punishers.

    It lives on the 2-simplex. The density `f(x, y)` of groups with fractions `x` of cooperators, `y` of defectors, and
    `1 - x - y` of punishers lives on the 2-simplex. The replicator dynamics within groups advect it, and groups win
    pairwise conflicts at the rate `group_strength f(x, y) int (rho(x, y; u, v) - rho(u, v; x, y)) f(u, v) du dv` over
    the simplex. The victory probability is `1/2 + (v - y) / 2` of the fraction of non-defectors (`'defectors'`), or the
    difference of the average payoffs normalized by their largest difference on the simplex (`'payoff'`) (Cooney 2024,
    arXiv 2405.18419 §3.2 and §7). The density starts uniform at two, so its integral is one, the cells along the
    slanted face are half cells, and no flux leaves the simplex.

    The system has the density `groups.x`.

    Args:
        cells:
            The number of cells of each axis of the simplex.

        victory:
            The victory probability, `'defectors'` or `'payoff'`.

        group_strength:
            The relative strength `lambda` of between-group conflict.

        benefit:
            The benefit `b` of the donation game.

        cost:
            The cost `c` of cooperating.

        punishment:
            The punishment `p` of a defector by every punisher it meets.

        interaction_cost:
            The cost `k` a punisher pays per defector it meets.

        fixed_cost:
            The fixed cost `q` of being able to punish.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import TrimorphicMultilevelSelectionSystem
        >>> TrimorphicMultilevelSelectionSystem(10).prepare().system_.elements[0].x.size
        100
    """

    field = 'social/cooperation'
    topics = ('field', 'multilevel selection', 'punishment', 'simplex', 'integral operator', 'pde')
    references = REFERENCES
    until = 30.0
    dt = 1.0

    @staticmethod
    def _get_trimorphic_payoff(x: Fraction, y: Fraction, payoffs: tuple[float, ...]) -> Fraction:
        """Return the average payoff `G(x, y)` of a group of three kinds.

        The kinds are cooperators `x`, defectors `y`, and punishers, and the payoff follows equation 2.5.
        """
        benefit, cost, punishment, interaction, fixed = payoffs
        strength = punishment + interaction
        return (
            benefit
            - cost
            - fixed
            + fixed * x
            - (benefit - cost - fixed + strength) * y
            + strength * x * y
            + strength * y**2
        )

    @staticmethod
    def _get_payoff_range(payoffs: tuple[float, ...]) -> float:
        """Return the largest difference `G* - G_*` of the average payoffs of two groups.

        The groups lie on the simplex.
        """
        grid = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
        x, y = np.meshgrid(grid, grid, indexing='ij')
        values = np.asarray(TrimorphicMultilevelSelectionSystem._get_trimorphic_payoff(x, y, payoffs))[x + y <= 1]
        return float(values.max() - values.min())

    def __init__(
        self,
        cells: int = 30,
        *,
        victory: str = 'defectors',
        group_strength: float = 4.0,
        benefit: float = 3.0,
        cost: float = 1.0,
        punishment: float = 0.5,
        interaction_cost: float = 0.1,
        fixed_cost: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.cells = cells
        self.victory = victory
        self.group_strength = group_strength
        self.benefit = benefit
        self.cost = cost
        self.punishment = punishment
        self.interaction_cost = interaction_cost
        self.fixed_cost = fixed_cost

    @staticmethod
    def _get_trimorphic_victory_gain(victory: str, payoffs: tuple[float, ...]) -> Expression:
        """Return `rho(x, y; u, v) - rho(u, v; x, y)` of the victory of a group.

        The victory is over another group on the simplex.
        """
        x, y = coordinate('x'), coordinate('y')
        u, v = integration_variable('x'), integration_variable('y')
        if victory == 'defectors':
            return v - y
        own, other = TrimorphicMultilevelSelectionSystem._get_trimorphic_payoff(
            x,
            y,
            payoffs,
        ), TrimorphicMultilevelSelectionSystem._get_trimorphic_payoff(u, v, payoffs)
        return (own - other) / TrimorphicMultilevelSelectionSystem._get_payoff_range(payoffs)

    @staticmethod
    def _get_simplex_drift(payoffs: tuple[float, ...]) -> list[Expression]:
        """Return the replicator dynamics of cooperators and defectors in a group.

        It follows equation E.20.
        """
        _, cost, punishment, interaction, fixed = payoffs
        x, y = coordinate('x'), coordinate('y')
        punished = fixed + interaction * y
        exploited = cost + fixed - punishment + punishment * x + (punishment + interaction) * y
        return [0, x * ((1 - x) * punished - y * exploited), y * ((1 - y) * exploited - x * punished)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        NonlinearPairwiseGroupConflictSystem._check_victory(self.victory, TRIMORPHIC_VICTORIES)
        payoffs = (self.benefit, self.cost, self.punishment, self.interaction_cost, self.fixed_cost)
        gain = self._get_trimorphic_victory_gain(self.victory, payoffs)
        axes = [AxisSpec('x', self.cells, 0.0, 1.0), AxisSpec('y', self.cells, 0.0, 1.0)]
        position = LatticeSpace((1,), discretization='finite-volume', scheme=FiniteVolumeSpec(limiter=None))
        space = SimplexSpace(position, axes)
        groups = GroupDensityElement('groups', x=2.0 * (space.prepare().volumes_ > 0), space=space)
        drift = self._get_simplex_drift(payoffs)
        within = SelectionDriftInteraction('within', target='groups', state='x', drift=lambda _: drift)
        between = Interaction(
            'between',
            roles={'target': 'groups'},
            laws=lambda s: [
                RateLaw(
                    s.target.x,
                    self.group_strength * s.target.x * integral_transform(s.target.x, gain, ('x', 'y')),
                ),
            ],
        )
        self.system_ = System(self.label, elements=[groups], interactions=[within, between], solver=RK4Solver(dt=0.01))
