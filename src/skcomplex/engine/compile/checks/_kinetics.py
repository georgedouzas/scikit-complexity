"""Check that the outcome probabilities of encounter laws sum to one over the outcomes of every encounter."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.laws import EncounterLaw, ProliferationLaw
from .._census import Census
from .._compilation import Compilation
from .._issues import Issue

PROBABILITY_TOLERANCE = 1e-9
LISTED_ENCOUNTERS = 3


def _probability_sum(total: object) -> float | None:
    """Return the sum of the outcome probabilities of one encounter as a number, or `None` when it holds a symbol."""
    value = SR(total).simplify_full() if isinstance(total, Expression) else SR(total)
    return float(value) if value.is_numeric() else None


def _unnormalized(law: EncounterLaw | ProliferationLaw) -> list[tuple[int, int, float]]:
    """Return the candidate and field activities of the encounters whose outcome probabilities do not sum to one."""
    table = law.transition if isinstance(law, EncounterLaw) else law.outcomes
    if table is None:
        return []
    sums = table.sum(axis=2)
    rates = np.broadcast_to(law.rate, sums.shape)
    found = []
    for (candidate, field_activity), total in np.ndenumerate(sums):
        rate = rates[candidate, field_activity]
        if SR(rate).is_numeric() and float(SR(rate)) == 0:
            continue
        value = _probability_sum(total)
        if value is not None and abs(value - 1) > PROBABILITY_TOLERANCE:
            found.append((candidate, field_activity, value))
    return found


def probability_issues(compilation: Compilation) -> list[Issue]:
    """Return the encounter laws whose outcome probabilities do not sum to one over the outcomes of an encounter.

    Args:
        compilation:
            The compilation state after planning.

    Returns:
        issues:
            The issues.
    """
    issues = []
    for record in cast('Census', compilation.census).records:
        law = record.law
        if not isinstance(law, EncounterLaw | ProliferationLaw):
            continue
        found = _unnormalized(law)
        if not found:
            continue
        listed = '; '.join(
            f'candidate activity {candidate} with field activity {field_activity} sums to {value:.6g}'
            for candidate, field_activity, value in found[:LISTED_ENCOUNTERS]
        )
        more = f', and {len(found) - LISTED_ENCOUNTERS} more' if len(found) > LISTED_ENCOUNTERS else ''
        message = (
            f'The {type(law).__name__} of {record.path} has outcome probabilities that do not sum to one over the '
            f'outcomes of an encounter: {listed}{more}.'
        )
        fix = 'Give outcome probabilities that sum to one over the outcomes of every encounter with a nonzero rate.'
        issues.append(Issue('encounter-probabilities', record.path, message, fix))
    return issues
