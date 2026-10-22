"""Lower encounter laws to the gain and loss rates of densities on activity grids, as sparse tensor contractions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError, symbols_of
from ..laws import Destruction, Encounter, Proliferation
from ._contractions import CompiledEncounter, Indices, TableValues
from ._kernels import Reference, Source

type EncounterLaw = Encounter | Proliferation | Destruction

ENCOUNTER_LAWS = (Encounter, Proliferation, Destruction)
KINDS = {Encounter: 'transition', Proliferation: 'proliferation', Destruction: 'destruction'}


def _nonzero(table: npt.NDArray[Any]) -> tuple[Indices, ...]:
    """Return the indices of the entries of a table that are not zero, numbers or Sage expressions."""
    if table.dtype != object:
        return tuple(np.asarray(item, dtype=np.intp) for item in np.nonzero(table))
    mask = np.vectorize(lambda entry: not (SR(entry).is_numeric() and SR(entry) == 0), otypes=[bool])(table)
    return tuple(np.asarray(item, dtype=np.intp) for item in np.nonzero(mask))


def _parameter_indices(
    table: npt.NDArray[Any],
    references: Mapping[Expression, Reference],
    owner: str,
) -> dict[Expression, int]:
    """Return the flat parameter index of every symbol of a table, after checking that the symbols are parameters."""
    if table.dtype != object:
        return {}
    indices = {}
    for item in sorted(symbols_of([SR(entry) for entry in table.ravel()]), key=str):
        reference = references.get(item)
        if reference is None or reference.is_state:
            error_msg = (
                f'An encounter law of {owner} reads {item} in its rates or outcome probabilities, which is not a '
                'parameter. Write encounter rates and probabilities with numbers and parameters.'
            )
            raise ValidationError(error_msg)
        indices[item] = reference.index
    return indices


def _group_indices(item: Expression | None, references: Mapping[Expression, Reference], owner: str) -> Indices:
    """Return the flat state indices of the density a symbol of an encounter stands for, one per activity."""
    reference = references.get(item) if item is not None else None
    if reference is None or not reference.is_state or reference.source is not Source.MEMBER:
        error_msg = (
            f'An encounter law of {owner} reads {item}, which is not a state of a group bound to a role. Encounters '
            'read the densities of groups on the sites of activity lattices, such as `s.candidate.f`.'
        )
        raise ValidationError(error_msg)
    return reference.flat_indices.astype(np.intp)


def _outcome_table(law: EncounterLaw, sizes: tuple[int, int, int], owner: str) -> npt.NDArray[Any] | None:
    """Return the outcome table of a law, the identity on the candidate's activity for a proliferation without one."""
    if isinstance(law, Destruction):
        return None
    table = law.transition if isinstance(law, Encounter) else law.outcomes
    if table is None:
        if sizes[0] != sizes[2]:
            error_msg = (
                f'A proliferation of {owner} has no outcome table, so the new particles keep the candidate activity, '
                f'but the test group has {sizes[2]} activities and the candidate {sizes[0]}. Give the outcome table.'
            )
            raise ValidationError(error_msg)
        identity = np.zeros(sizes)
        identity[np.arange(sizes[0]), :, np.arange(sizes[0])] = 1.0
        return identity
    if table.shape != sizes:
        error_msg = (
            f'An encounter law of {owner} has an outcome table of shape {table.shape}, but its candidate, field, and '
            f'test groups need the shape {sizes}.'
        )
        raise ValidationError(error_msg)
    return table


def _check_rate(law: EncounterLaw, sizes: tuple[int, int, int], owner: str) -> None:
    """Raise an error when the rate table of a law does not have one row per candidate and one column per field."""
    if law.rate.ndim and law.rate.shape != sizes[:2]:
        error_msg = (
            f'An encounter law of {owner} has a rate of shape {law.rate.shape}, but its candidate and field groups '
            f'need the shape {sizes[:2]}.'
        )
        raise ValidationError(error_msg)


def compile_encounter(
    law: EncounterLaw,
    owner: str,
    references: Mapping[Expression, Reference],
    time: Expression,
    *,
    fused: bool = False,
) -> CompiledEncounter:
    """Return an encounter law lowered to sparse contractions over the densities its roles bind.

    Args:
        law:
            The encounter, proliferation, or destruction law.

        owner:
            The path of the component that returns the law.

        references:
            The references of the symbols of the component's namespace.

        time:
            The time symbol.

        fused:
            Whether the gain runs in one fused loop compiled with Numba.

    Returns:
        encounter:
            The compiled law.

    Raises:
        ValidationError:
            A density is not a state of a group, a table has the wrong shape, or a table reads a state.
    """
    candidate = _group_indices(law.candidate, references, owner)
    field_indices = _group_indices(law.field, references, owner)
    test_symbol = getattr(law, 'test', None)
    test = candidate if test_symbol is None else _group_indices(test_symbol, references, owner)
    sizes = (candidate.size, field_indices.size, test.size)
    _check_rate(law, sizes, owner)
    table = _outcome_table(law, sizes, owner)
    grid = tuple(item.ravel().astype(np.intp) for item in np.indices(sizes[:2]))
    rate = TableValues(law.rate, grid, _parameter_indices(law.rate, references, owner), time)
    if table is None:
        empty = np.zeros(0, dtype=np.intp)
        return CompiledEncounter(owner, KINDS[type(law)], candidate, field_indices, test, rate, (empty,) * 3, None)
    positions = _nonzero(table)
    entries = (positions[0], positions[1], positions[2])
    probabilities = TableValues(table, positions, _parameter_indices(table, references, owner), time)
    return CompiledEncounter(
        owner,
        KINDS[type(law)],
        candidate,
        field_indices,
        test,
        rate,
        entries,
        probabilities,
        fused,
    )


def compile_encounters(
    components: Sequence[tuple[str, Mapping[Expression, Reference], Sequence[object]]],
    time: Expression,
    *,
    fused: bool = False,
) -> tuple[CompiledEncounter, ...]:
    """Return the encounter laws of every component, lowered in declaration order.

    Args:
        components:
            The path, the references of the namespace, and the laws of every component.

        time:
            The time symbol.

        fused:
            Whether the gains run in fused loops compiled with Numba.

    Returns:
        encounters:
            The compiled laws.
    """
    return tuple(
        compile_encounter(law, owner, references, time, fused=fused)
        for owner, references, laws in components
        for law in laws
        if isinstance(law, ENCOUNTER_LAWS)
    )
