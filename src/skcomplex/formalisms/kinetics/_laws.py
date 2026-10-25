"""Define the encounter laws of kinetic models, which move, create, and remove particles over activity grids."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self, TypeGuard

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ...core.base import (
    BaseLaw,
    DomainKind,
    Expression,
    LawClass,
    LawKind,
    Requires,
    SpaceKind,
    TimeKind,
    Trait,
    ValidationError,
)

ENCOUNTER_REQUIREMENTS = Requires(time={TimeKind.CONTINUOUS}, space={SpaceKind.LATTICE}, domains={DomainKind.REAL})

type Table = npt.NDArray[Any]
type KineticLaw = EncounterLaw | ProliferationLaw | DestructionLaw


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


def is_encounter(law: BaseLaw) -> TypeGuard[KineticLaw]:
    """Check whether a law is an encounter law, a law of the encounter kind."""
    return law.law_kind == LawKind.ENCOUNTER


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
        >>> from skcomplex.formalisms.kinetics import outcome_sums
        >>> outcome_sums(np.full((2, 2, 2), 0.5)).tolist()
        [[1.0, 1.0], [1.0, 1.0]]
    """
    return outcomes.sum(axis=2)


def _encounter_targets(law: 'EncounterLaw | ProliferationLaw') -> tuple[Expression, ...]:
    """Return the densities an encounter changes, the candidate, whose group owns the law, and the test."""
    if law.test is None or law.test is law.candidate:
        return (law.candidate,)
    return (law.candidate, law.test)


@dataclass(frozen=True, eq=False)
class EncounterLaw(BaseLaw):
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
        >>> from skcomplex.formalisms.kinetics import EncounterLaw
        >>> f, g = symbol('f'), symbol('g')
        >>> EncounterLaw(f, g, 1.0, np.full((2, 2, 2), 0.5)).check().transition.shape
        (2, 2, 2)
    """

    law_kind: ClassVar[LawKind] = LawKind.ENCOUNTER
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.FORMLESS, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = ENCOUNTER_REQUIREMENTS
    effect: ClassVar[str] = 'transition'

    candidate: Expression
    field: Expression
    rate: Any
    transition: Any
    test: Expression | None = None

    def _check(self) -> Self:
        """Return the law with its rate and transition table as arrays, raising an error for invalid tables."""
        description = f'encounter of {self.candidate} with {self.field}'
        rate = _table(self.rate, (0, 2), f'rate of the {description}')
        transition = _table(self.transition, (3,), f'transition table of the {description}')
        _check_sizes(rate, transition, description)
        return replace(self, rate=rate, transition=transition, targets=_encounter_targets(self))

    @property
    def outcomes(self) -> Table:
        """Table of the activities that particles end at, the transition probabilities."""
        return self.transition


@dataclass(frozen=True, eq=False)
class ProliferationLaw(BaseLaw):
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
        >>> from skcomplex.formalisms.kinetics import ProliferationLaw
        >>> f, g = symbol('f'), symbol('g')
        >>> ProliferationLaw(f, g, 0.5).check().outcomes is None
        True
    """

    law_kind: ClassVar[LawKind] = LawKind.ENCOUNTER
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.FORMLESS, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = ENCOUNTER_REQUIREMENTS
    effect: ClassVar[str] = 'proliferation'

    candidate: Expression
    field: Expression
    rate: Any
    outcomes: Any = None
    test: Expression | None = None

    def _check(self) -> Self:
        """Return the law with its rate and outcome table as arrays, raising an error for invalid tables."""
        description = f'proliferation of {self.candidate} with {self.field}'
        rate = _table(self.rate, (0, 2), f'rate of the {description}')
        outcomes = None if self.outcomes is None else _table(self.outcomes, (3,), f'outcome table of the {description}')
        _check_sizes(rate, outcomes, description)
        return replace(self, rate=rate, outcomes=outcomes, targets=_encounter_targets(self))


@dataclass(frozen=True, eq=False)
class DestructionLaw(BaseLaw):
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
        >>> from skcomplex.formalisms.kinetics import DestructionLaw
        >>> f, g = symbol('f'), symbol('g')
        >>> float(DestructionLaw(f, g, 0.2).check().rate)
        0.2
    """

    law_kind: ClassVar[LawKind] = LawKind.ENCOUNTER
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.FORMLESS, Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = ENCOUNTER_REQUIREMENTS
    effect: ClassVar[str] = 'destruction'

    candidate: Expression
    field: Expression
    rate: Any

    def _check(self) -> Self:
        """Return the law with its rate as an array, raising an error for an invalid rate."""
        rate = _table(self.rate, (0, 2), f'rate of the destruction of {self.candidate} by {self.field}')
        return replace(self, rate=rate, targets=(self.candidate,))
