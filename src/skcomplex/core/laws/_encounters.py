"""Define the encounter laws of kinetic models, which move, create, and remove particles over activity grids."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import DomainKind, Expression, LawKind, Requires, SpaceKind, TimeKind, ValidationError

ENCOUNTER_REQUIREMENTS = Requires(time={TimeKind.CONTINUOUS}, space={SpaceKind.LATTICE}, domains={DomainKind.REAL})

type Table = npt.NDArray[Any]


def _table(value: object, dimensions: tuple[int, ...], description: str) -> Table:
    """Return a table as a float array, or as an object array of Sage expressions when an entry holds a symbol."""
    array = value if isinstance(value, np.ndarray) else np.asarray(value, dtype=object)
    if array.ndim not in dimensions:
        expected = ' or '.join(str(item) for item in dimensions)
        error_msg = f'The {description} must be an array of {expected} dimensions, but got shape {array.shape}.'
        raise ValidationError(error_msg)
    try:
        return np.asarray(array, dtype=float)
    except TypeError:
        return np.vectorize(SR, otypes=[object])(array)


def _check_sizes(rate: Table, outcomes: Table | None, description: str) -> None:
    """Raise an error when the rate table and the outcome table disagree on the candidate and field activities."""
    if outcomes is None or rate.ndim == 0 or rate.shape == outcomes.shape[:2]:
        return
    error_msg = (
        f'The rate of the {description} has the shape {rate.shape}, but its outcome table has '
        f'{outcomes.shape[:2]} candidate and field activities.'
    )
    raise ValidationError(error_msg)


def outcome_sums(outcomes: Table) -> Table:
    """Return the sum of an outcome table over the outcomes of every encounter.

    Args:
        outcomes:
            The table `A[h, k, i]` of the probability that a candidate at activity `h` that meets a field particle at
            activity `k` ends at activity `i`.

    Returns:
        sums:
            The table of `sum_i A[h, k, i]`, of floats or Sage expressions.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.laws import outcome_sums
        >>> outcome_sums(np.full((2, 2, 2), 0.5)).tolist()
        [[1.0, 1.0], [1.0, 1.0]]
    """
    return outcomes.sum(axis=2)


@dataclass(frozen=True, init=False, eq=False)
class Encounter:
    """Conservative encounter of active particles, which moves a candidate particle to a new activity.

    A candidate particle at activity `h` meets a field particle at activity `k` at the encounter rate `eta[h, k]`, and
    ends at activity `i` with the transition probability `A[h, k, i]`. With candidate density `f` and field density
    `g` on activity grids, the law adds Bellomo's gain and loss terms to the rate of the test density,

    `df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`,

    where the loss acts on the candidate and the gain on the test group, the candidate's own group by default. The
    candidate, the field, and the test are states of groups on the sites of activity lattices, one member per
    activity. A field role bound to a group whose states no law changes is an external action at the micro scale.

    Args:
        candidate:
            The symbol of the candidate density, a state of a group read member by member, such as `s.wealth.f`.

        field:
            The symbol of the field density.

        rate:
            The encounter rate, a number or a parameter expression, or an array `eta[h, k]` of them.

        transition:
            The array `A[h, k, i]` of transition probabilities, numbers or parameter expressions, which sum to one
            over `i`.

        test:
            The symbol of the density the candidate moves into, or `None` for the candidate's own density.

    Raises:
        ValidationError:
            The rate or the transition table has the wrong number of dimensions, or their activities disagree.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Encounter
        >>> f, g = symbol('f'), symbol('g')
        >>> Encounter(f, g, 1.0, np.full((2, 2, 2), 0.5)).transition.shape
        (2, 2, 2)
    """

    law_kind: ClassVar[LawKind] = LawKind.ENCOUNTER
    requires: ClassVar[Requires] = ENCOUNTER_REQUIREMENTS

    candidate: Expression
    field: Expression
    rate: Table
    transition: Table
    test: Expression | None

    def __init__(
        self,
        candidate: Expression,
        field: Expression,
        rate: object,
        transition: object,
        test: Expression | None = None,
    ) -> None:
        description = f'encounter of {candidate} with {field}'
        object.__setattr__(self, 'candidate', candidate)
        object.__setattr__(self, 'field', field)
        object.__setattr__(self, 'rate', _table(rate, (0, 2), f'rate of the {description}'))
        object.__setattr__(self, 'transition', _table(transition, (3,), f'transition table of the {description}'))
        object.__setattr__(self, 'test', test)
        _check_sizes(self.rate, self.transition, description)

    @property
    def target(self) -> Expression:
        """Symbol of the candidate density, whose group owns the law."""
        return self.candidate

    @property
    def outcomes(self) -> Table:
        """Table of the activities that particles end at, the transition probabilities."""
        return self.transition


@dataclass(frozen=True, init=False, eq=False)
class Proliferation:
    """Proliferative encounter, which creates a particle in the test group without removing the candidate.

    A candidate particle at activity `h` that meets a field particle at activity `k` creates, at the rate `mu[h, k]`,
    a new particle at activity `i` with probability `B[h, k, i]`, so the law adds
    `sum_hk mu[h, k] B[h, k, i] f_h g_k` to the rate of the test density at `i`.

    Args:
        candidate:
            The symbol of the candidate density.

        field:
            The symbol of the field density.

        rate:
            The proliferation rate, a number or a parameter expression, or an array `mu[h, k]` of them.

        outcomes:
            The array `B[h, k, i]` of the probabilities of the new particle's activity, or `None` for the
            candidate's own activity, which needs a test group of the candidate's size.

        test:
            The symbol of the density that gains the new particles, or `None` for the candidate's own density.

    Raises:
        ValidationError:
            The rate or the outcome table has the wrong number of dimensions, or their activities disagree.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Proliferation
        >>> f, g = symbol('f'), symbol('g')
        >>> Proliferation(f, g, 0.5).outcomes is None
        True
    """

    law_kind: ClassVar[LawKind] = LawKind.ENCOUNTER
    requires: ClassVar[Requires] = ENCOUNTER_REQUIREMENTS

    candidate: Expression
    field: Expression
    rate: Table
    outcomes: Table | None
    test: Expression | None

    def __init__(
        self,
        candidate: Expression,
        field: Expression,
        rate: object,
        outcomes: object = None,
        test: Expression | None = None,
    ) -> None:
        description = f'proliferation of {candidate} with {field}'
        table = None if outcomes is None else _table(outcomes, (3,), f'outcome table of the {description}')
        object.__setattr__(self, 'candidate', candidate)
        object.__setattr__(self, 'field', field)
        object.__setattr__(self, 'rate', _table(rate, (0, 2), f'rate of the {description}'))
        object.__setattr__(self, 'outcomes', table)
        object.__setattr__(self, 'test', test)
        _check_sizes(self.rate, self.outcomes, description)

    @property
    def target(self) -> Expression:
        """Symbol of the candidate density, whose group owns the law."""
        return self.candidate


@dataclass(frozen=True, init=False, eq=False)
class Destruction:
    """Destructive encounter, which removes the candidate particle.

    A candidate particle at activity `h` that meets a field particle at activity `k` is removed at the rate
    `mu[h, k]`, so the law adds `-f_h sum_k mu[h, k] g_k` to the rate of the candidate density at `h`.

    Args:
        candidate:
            The symbol of the candidate density.

        field:
            The symbol of the field density.

        rate:
            The destruction rate, a number or a parameter expression, or an array `mu[h, k]` of them.

    Raises:
        ValidationError:
            The rate has the wrong number of dimensions.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Destruction
        >>> f, g = symbol('f'), symbol('g')
        >>> float(Destruction(f, g, 0.2).rate)
        0.2
    """

    law_kind: ClassVar[LawKind] = LawKind.ENCOUNTER
    requires: ClassVar[Requires] = ENCOUNTER_REQUIREMENTS

    candidate: Expression
    field: Expression
    rate: Table

    def __init__(self, candidate: Expression, field: Expression, rate: object) -> None:
        object.__setattr__(self, 'candidate', candidate)
        object.__setattr__(self, 'field', field)
        object.__setattr__(self, 'rate', _table(rate, (0, 2), f'rate of the destruction of {candidate} by {field}'))

    @property
    def target(self) -> Expression:
        """Symbol of the candidate density, whose group owns the law."""
        return self.candidate


type EncounterLaw = Encounter | Proliferation | Destruction
