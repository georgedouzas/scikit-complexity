"""Multilevel selection of altruistic punishment: densities of group compositions under pairwise group conflict."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, tanh

from ....core.components.elements import DensityElement, FieldElement
from ....core.components.interactions import AdvectionInteraction, RateInteraction
from ....core.components.laws.utils import coordinate, integral_transform, integration_variable
from ....core.components.spaces import LatticeSpace, PhaseSpace, SimplexSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.errors import ValidationError
from ....core.specs import AxisSpec, VisualSpec
from ..._catalog import register_system

PAIRWISE_VICTORIES = ('fermi', 'local', 'tullock')
TRIMORPHIC_VICTORIES = ('defectors', 'payoff')
SIMPLEX_SAMPLES = 401
REFERENCES = ('Cooney 2024',)
SIMPLEX_VISUAL = VisualSpec('image', {'slices': {0: 0}})

type Fraction = Expression | npt.NDArray[np.float64]


def _get_pairwise_payoff(z: Fraction, payoffs: tuple[float, ...]) -> Fraction:
    """Return the average payoff `G(z)` of a group of punishers `z` and defectors, from equation 2.11."""
    benefit, cost, punishment, interaction, fixed = payoffs
    return (benefit - cost - fixed - punishment - interaction) * z + (punishment + interaction) * z**2


def _get_trimorphic_payoff(x: Fraction, y: Fraction, payoffs: tuple[float, ...]) -> Fraction:
    """Return the average payoff `G(x, y)` of cooperators `x`, defectors `y`, and punishers, from equation 2.5."""
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


def _get_pairwise_victory(
    first: Expression,
    second: Expression,
    payoffs: tuple[float, ...],
    *,
    victory: str,
    sensitivity: float,
    exponent: float,
) -> Expression:
    """Return the probability that a group of punishers `first` defeats one of `second`, from equation 3.12 to 3.14."""
    own, other = (_get_pairwise_payoff(item, payoffs) for item in (first, second))
    if victory == 'fermi':
        return (1 + tanh(sensitivity * (own - other))) / 2
    if victory == 'local':
        return (1 + (own - other) / (abs(own) + abs(other))) / 2
    centres = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
    lowest = float(np.min(np.asarray(_get_pairwise_payoff(centres, payoffs), dtype=float)))
    own_share, other_share = (own - lowest) ** (1 / exponent), (other - lowest) ** (1 / exponent)
    return own_share / (own_share + other_share)


def _get_payoff_range(payoffs: tuple[float, ...]) -> float:
    """Return the largest difference `G* - G_*` of the average payoffs of two groups on the simplex."""
    grid = np.linspace(0.0, 1.0, SIMPLEX_SAMPLES)
    x, y = np.meshgrid(grid, grid, indexing='ij')
    values = np.asarray(_get_trimorphic_payoff(x, y, payoffs))[x + y <= 1]
    return float(values.max() - values.min())


def _get_trimorphic_victory_gain(victory: str, payoffs: tuple[float, ...]) -> Expression:
    """Return `rho(x, y; u, v) - rho(u, v; x, y)` of the victory of a group over another on the simplex."""
    x, y = coordinate('x'), coordinate('y')
    u, v = integration_variable('x'), integration_variable('y')
    if victory == 'defectors':
        return v - y
    own, other = _get_trimorphic_payoff(x, y, payoffs), _get_trimorphic_payoff(u, v, payoffs)
    return (own - other) / _get_payoff_range(payoffs)


def _get_simplex_drift(payoffs: tuple[float, ...]) -> list[Expression]:
    """Return the replicator dynamics of cooperators and defectors in a group, from equation E.20."""
    _, cost, punishment, interaction, fixed = payoffs
    x, y = coordinate('x'), coordinate('y')
    punished = fixed + interaction * y
    exploited = cost + fixed - punishment + punishment * x + (punishment + interaction) * y
    return [0, x * ((1 - x) * punished - y * exploited), y * ((1 - y) * exploited - x * punished)]


def _check_victory(victory: str, victories: tuple[str, ...]) -> None:
    """Raise an error for a victory probability that is not one of the given ones."""
    if victory not in victories:
        error_msg = f'Unknown victory probability {victory!r}. Use one of {", ".join(victories)}.'
        raise ValidationError(error_msg)


@register_system(
    field='social/cooperation',
    tags=('field', 'multilevel selection', 'punishment', 'integral operator', 'pde'),
    references=REFERENCES,
    defaults={'until': 15.0, 'dt': 0.5, 'solver': RK4Solver(dt=0.01)},
)
def nonlinear_pairwise_group_conflict(
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
) -> System:
    """Return the density of groups of defectors and altruistic punishers under pairwise group conflict.

    The density `f(z)` of groups with a fraction `z` of punishers is advected by the replicator dynamics within groups,
    `z (1 - z) (pi_P - pi_D)`, and groups win pairwise conflicts, copying themselves over the loser, at the rate
    `group_strength f(z) int (rho(z, u) - rho(u, z)) f(u) du`. The victory probability `rho` is not additively
    separable: the group Fermi rule `(1 + tanh(s (G(z) - G(u)))) / 2` (`'fermi'`), the local rule normalized by the
    payoffs of both groups (`'local'`), or the Tullock contest function of the payoffs above their minimum with the
    exponent `1 / a` (`'tullock'`), where `G(z)` is the average payoff of a group (Cooney 2024, arXiv 2405.18419 §3.1
    and §6). The density starts uniform, and the integral against the two-point kernel `rho(z, u) - rho(u, z)` keeps
    its total at one.

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

    Returns:
        system:
            The system, with the density `groups.f`.

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import nonlinear_pairwise_group_conflict
        >>> nonlinear_pairwise_group_conflict(40).elements[0].prepare().size_
        40
    """
    _check_victory(victory, PAIRWISE_VICTORIES)
    payoffs = (benefit, cost, punishment, interaction_cost, fixed_cost)
    z, u = coordinate('z'), integration_variable('z')
    win = _get_pairwise_victory(z, u, payoffs, victory=victory, sensitivity=sensitivity, exponent=exponent)
    loss = _get_pairwise_victory(u, z, payoffs, victory=victory, sensitivity=sensitivity, exponent=exponent)
    gain = win - loss
    advantage = (punishment + interaction_cost) * z - (cost + fixed_cost + interaction_cost)
    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', cells, 0.0, 1.0)])
    groups = FieldElement(
        'groups',
        element=DensityElement,
        space=space,
        values={'f': np.ones((1, cells))},
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
        rate=lambda s: group_strength * s.target.f * integral_transform(s.target.f, gain, ('z',)),
    )
    return System(elements=[groups], interactions=[within, between])


@register_system(
    field='social/cooperation',
    tags=('field', 'multilevel selection', 'punishment', 'simplex', 'integral operator', 'pde'),
    references=REFERENCES,
    defaults={'until': 30.0, 'dt': 1.0, 'solver': RK4Solver(dt=0.01)},
)
def trimorphic_multilevel_selection(
    cells: int = 30,
    *,
    victory: str = 'defectors',
    group_strength: float = 4.0,
    benefit: float = 3.0,
    cost: float = 1.0,
    punishment: float = 0.5,
    interaction_cost: float = 0.1,
    fixed_cost: float = 0.0,
) -> System:
    """Return the density of groups of cooperators, defectors, and altruistic punishers on the 2-simplex.

    The density `f(x, y)` of groups with fractions `x` of cooperators, `y` of defectors, and `1 - x - y` of punishers
    lives on the 2-simplex. The replicator dynamics within groups advect it, and groups win pairwise conflicts at the
    rate `group_strength f(x, y) int (rho(x, y; u, v) - rho(u, v; x, y)) f(u, v) du dv` over the simplex. The victory
    probability is `1/2 + (v - y) / 2` of the fraction of non-defectors (`'defectors'`), or the difference of the
    average payoffs normalized by their largest difference on the simplex (`'payoff'`) (Cooney 2024, arXiv 2405.18419
    §3.2 and §7). The density starts uniform at two, so its integral is one, the cells along the slanted face are half
    cells, and no flux leaves the simplex.

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

    Returns:
        system:
            The system, with the density `groups.f`.

    Raises:
        ValidationError:
            The victory probability is unknown.

    Examples:
        >>> from skcomplex.systems.social.cooperation import trimorphic_multilevel_selection
        >>> trimorphic_multilevel_selection(10).elements[0].prepare().size_
        100
    """
    _check_victory(victory, TRIMORPHIC_VICTORIES)
    payoffs = (benefit, cost, punishment, interaction_cost, fixed_cost)
    gain = _get_trimorphic_victory_gain(victory, payoffs)
    axes = [AxisSpec('x', cells, 0.0, 1.0), AxisSpec('y', cells, 0.0, 1.0)]
    space = SimplexSpace(LatticeSpace((1,)), axes)
    groups = FieldElement(
        'groups',
        element=DensityElement,
        space=space,
        values={'f': 2.0 * (space.prepare().volumes_ > 0)},
        discretization='finite-volume',
        visual=SIMPLEX_VISUAL,
    )
    drift = _get_simplex_drift(payoffs)
    within = AdvectionInteraction('within', target='groups', state='f', drift=lambda _: drift, limiter=None)
    between = RateInteraction(
        'between',
        target='groups',
        state='f',
        rate=lambda s: group_strength * s.target.f * integral_transform(s.target.f, gain, ('x', 'y')),
    )
    return System(elements=[groups], interactions=[within, between])
