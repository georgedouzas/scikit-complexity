"""Multilevel selection of altruistic punishment.

It follows the densities of group compositions under group-level competition.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, tanh

from ....core.components.elements import DensityElement, FieldElement
from ....core.components.interactions import AdvectionInteraction, RateInteraction
from ....core.components.laws.utils import coordinate, integral_transform, integration_variable
from ....core.components.spaces import LatticeSpace, PhaseSpace, SimplexSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import RK4Solver
from ....core.errors import ValidationError
from ....core.specs import AxisSpec, VisualSpec

PAIRWISE_VICTORIES = ('fermi', 'local', 'tullock')
TRIMORPHIC_VICTORIES = ('defectors', 'payoff')
ADDITIVE_VICTORIES = ('punishers', 'payoff')
SIMPLEX_SAMPLES = 401
REFERENCES = ('Cooney 2024',)
SIMPLEX_VISUAL = VisualSpec('image', {'slices': {0: 0}})

type Fraction = Expression | npt.NDArray[np.float64]


def _get_pairwise_payoff(z: Fraction, payoffs: tuple[float, ...]) -> Fraction:
    """Return the average payoff `G(z)` of a group of punishers `z` and defectors.

    It follows equation 2.11.
    """
    benefit, cost, punishment, interaction, fixed = payoffs
    return (benefit - cost - fixed - punishment - interaction) * z + (punishment + interaction) * z**2


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


def _get_payoff_range(payoffs: tuple[float, ...]) -> float:
    """Return the largest difference `G* - G_*` of the average payoffs of two groups.

    The groups lie on the simplex.
    """
    grid = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
    x, y = np.meshgrid(grid, grid, indexing='ij')
    values = np.asarray(_get_trimorphic_payoff(x, y, payoffs))[x + y <= 1]
    return float(values.max() - values.min())


def _check_victory(victory: str, victories: tuple[str, ...]) -> None:
    """Raise an error for a victory probability that is not one of the given ones."""
    if victory not in victories:
        error_msg = f'Unknown victory probability {victory!r}. Use one of {", ".join(victories)}.'
        raise ValidationError(error_msg)


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

    The system has the density `groups.f`.

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

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import NonlinearPairwiseGroupConflictSystem
        >>> NonlinearPairwiseGroupConflictSystem(40).prepare().system_.elements[0].prepare().size_
        40
    """

    field = 'social/cooperation'
    topics = ('field', 'multilevel selection', 'punishment', 'integral operator', 'pde')
    references = REFERENCES
    until = 15.0
    dt = 0.5
    solver = RK4Solver(dt=0.01)

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
    ) -> None:
        super().__init__(label)
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
        own, other = (_get_pairwise_payoff(item, payoffs) for item in (first, second))
        if victory == 'fermi':
            return (1 + tanh(sensitivity * (own - other))) / 2
        if victory == 'local':
            return (1 + (own - other) / (abs(own) + abs(other))) / 2
        centres = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
        lowest = float(np.min(np.asarray(_get_pairwise_payoff(centres, payoffs), dtype=float)))
        own_share, other_share = (own - lowest) ** (1 / exponent), (other - lowest) ** (1 / exponent)
        return own_share / (own_share + other_share)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        _check_victory(self.victory, PAIRWISE_VICTORIES)
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
        space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', self.cells, 0.0, 1.0)])
        groups = FieldElement(
            'groups',
            element=DensityElement,
            space=space,
            values={'f': np.ones((1, self.cells))},
            discretization='finite-volume',
        )
        within = AdvectionInteraction(
            'within',
            target='groups',
            state='f',
            drift=lambda _: [0, z * (1 - z) * advantage],
            limiter=None,
        )
        between = RateInteraction(
            'between',
            target='groups',
            state='f',
            rate=lambda s: self.group_strength * s.target.f * integral_transform(s.target.f, gain, ('z',)),
        )
        return System(self.label, elements=[groups], interactions=[within, between])


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

    The system has the density `groups.f`.

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

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import TwoLevelReplicatorSystem
        >>> TwoLevelReplicatorSystem(40).prepare().system_.elements[0].prepare().size_
        40
    """

    field = 'social/cooperation'
    topics = ('field', 'multilevel selection', 'punishment', 'two-level replicator', 'pde')
    references = REFERENCES
    until = 40.0
    dt = 1.0
    solver = RK4Solver(dt=0.01)

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
    ) -> None:
        super().__init__(label)
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
        values = np.asarray(_get_pairwise_payoff(centres, payoffs), dtype=float)
        return _get_pairwise_payoff(z, payoffs) / float(values.max() - values.min())

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        _check_victory(self.victory, ADDITIVE_VICTORIES)
        payoffs = (self.benefit, self.cost, self.punishment, self.interaction_cost, self.fixed_cost)
        z, u = coordinate('z'), integration_variable('z')
        gain = self._get_group_rate(z, payoffs, self.victory) - self._get_group_rate(u, payoffs, self.victory)
        advantage = (self.punishment + self.interaction_cost) * z - (
            self.cost + self.fixed_cost + self.interaction_cost
        )
        space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', self.cells, 0.0, 1.0)])
        groups = FieldElement(
            'groups',
            element=DensityElement,
            space=space,
            values={'f': np.ones((1, self.cells))},
            discretization='finite-volume',
        )
        within = AdvectionInteraction(
            'within',
            target='groups',
            state='f',
            drift=lambda _: [0, z * (1 - z) * advantage],
            limiter=None,
        )
        between = RateInteraction(
            'between',
            target='groups',
            state='f',
            rate=lambda s: self.group_strength * s.target.f * integral_transform(s.target.f, gain, ('z',)),
        )
        return System(self.label, elements=[groups], interactions=[within, between])


class TrimorphicMultilevelSelectionSystem(BaseSystem):
    """The density of groups of cooperators, defectors, and altruistic punishers.

    It lives on the 2-simplex. The density `f(x, y)` of groups with fractions `x` of cooperators, `y` of defectors, and
    `1 - x - y` of punishers lives on the 2-simplex. The replicator dynamics within groups advect it, and groups win
    pairwise conflicts at the rate `group_strength f(x, y) int (rho(x, y; u, v) - rho(u, v; x, y)) f(u, v) du dv` over
    the simplex. The victory probability is `1/2 + (v - y) / 2` of the fraction of non-defectors (`'defectors'`), or the
    difference of the average payoffs normalized by their largest difference on the simplex (`'payoff'`) (Cooney 2024,
    arXiv 2405.18419 §3.2 and §7). The density starts uniform at two, so its integral is one, the cells along the
    slanted face are half cells, and no flux leaves the simplex.

    The system has the density `groups.f`.

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

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import TrimorphicMultilevelSelectionSystem
        >>> TrimorphicMultilevelSelectionSystem(10).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'social/cooperation'
    topics = ('field', 'multilevel selection', 'punishment', 'simplex', 'integral operator', 'pde')
    references = REFERENCES
    until = 30.0
    dt = 1.0
    solver = RK4Solver(dt=0.01)

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
    ) -> None:
        super().__init__(label)
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
        own, other = _get_trimorphic_payoff(x, y, payoffs), _get_trimorphic_payoff(u, v, payoffs)
        return (own - other) / _get_payoff_range(payoffs)

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

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        _check_victory(self.victory, TRIMORPHIC_VICTORIES)
        payoffs = (self.benefit, self.cost, self.punishment, self.interaction_cost, self.fixed_cost)
        gain = self._get_trimorphic_victory_gain(self.victory, payoffs)
        axes = [AxisSpec('x', self.cells, 0.0, 1.0), AxisSpec('y', self.cells, 0.0, 1.0)]
        space = SimplexSpace(LatticeSpace((1,)), axes)
        groups = FieldElement(
            'groups',
            element=DensityElement,
            space=space,
            values={'f': 2.0 * (space.prepare().volumes_ > 0)},
            discretization='finite-volume',
            visual=SIMPLEX_VISUAL,
        )
        drift = self._get_simplex_drift(payoffs)
        within = AdvectionInteraction('within', target='groups', state='f', drift=lambda _: drift, limiter=None)
        between = RateInteraction(
            'between',
            target='groups',
            state='f',
            rate=lambda s: self.group_strength * s.target.f * integral_transform(s.target.f, gain, ('x', 'y')),
        )
        return System(self.label, elements=[groups], interactions=[within, between])
