"""Pass that compiles the encounter laws before reductions prune the groups they read, and takes them out."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ....core.laws import BaseLaw, DestructionLaw, EncounterLaw, ProliferationLaw
from ....errors import ValidationError
from ....specs import ReferenceSpec, TableValuesSpec
from ....types import Indices, ReadKind
from ..._utils import symbols_of
from ...backends import BaseBackend
from ...problems._encounter_kernel import EncounterKernel
from .._base import BaseCompiler, BasePass
from .._compilation import Laws
from .._declarations import Declarations

_KINETIC_LAWS = (EncounterLaw, ProliferationLaw, DestructionLaw)

type KineticLaw = EncounterLaw | ProliferationLaw | DestructionLaw


class CompileEncountersPass(BasePass):
    """Pass that compiles the encounter laws before reductions prune the groups they read, and takes them out."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn', 'encounters')

    def _run(self, inputs: object = None) -> None:
        """Compile the encounter laws of the drawn laws."""
        compilation = self.compilation_
        specs = cast('Laws', compilation.drawn)
        declarations = cast('Declarations', compilation.layout)
        compilation.encounters = self._compiled_encounters(specs, declarations, compilation.backend)
        compilation.drawn = [
            (spec, [law for law in laws if not isinstance(law, _KINETIC_LAWS)]) for spec, laws in specs
        ]

    @staticmethod
    def _compiled_encounters(
        specs: Laws,
        declarations: Declarations,
        backend: BaseBackend | None,
    ) -> tuple[EncounterKernel, ...]:
        """Return the encounter laws of every component, lowered for their members."""
        if not any(isinstance(law, _KINETIC_LAWS) for _, laws in specs for law in laws):
            return ()
        members = Declarations._member_count(list(declarations.blocks.values()))
        fused = BaseCompiler._resolve_backend(backend, members)._get_name() == 'numba'
        components = [(spec.owner, dict(spec.references), laws) for spec, laws in specs]
        return CompileEncountersPass._compile_encounters(components, declarations.time, fused=fused)

    @staticmethod
    def _effect(law: KineticLaw) -> str:
        """Return what an encounter law does to particles, `'transition'`, `'proliferation'`, or `'destruction'`."""
        if isinstance(law, EncounterLaw):
            return 'transition'
        return 'proliferation' if isinstance(law, ProliferationLaw) else 'destruction'

    @staticmethod
    def _nonzero(table: npt.NDArray[Any]) -> tuple[Indices, ...]:
        """Return the indices of the entries of a table that are not zero, numbers or Sage expressions."""
        if table.dtype != object:
            return tuple(np.asarray(item, dtype=np.intp) for item in np.nonzero(table))
        mask = np.vectorize(lambda entry: not (SR(entry).is_numeric() and SR(entry) == 0), otypes=[bool])(table)
        return tuple(np.asarray(item, dtype=np.intp) for item in np.nonzero(mask))

    @staticmethod
    def _group_indices(item: Expression | None, references: Mapping[Expression, ReferenceSpec], owner: str) -> Indices:
        """Return the flat state indices of the density a symbol of an encounter stands for, one per activity."""
        reference = references.get(item) if item is not None else None
        if reference is None or not reference.is_state or reference.source is not ReadKind.MEMBER:
            error_msg = (
                f'An encounter law of {owner} reads {item}, which is not a state of a group bound to a role. '
                'Encounters read the densities of groups on the sites of activity lattices, such as `s.candidate.f`.'
            )
            raise ValidationError(error_msg)
        return reference.flat_indices.astype(np.intp)

    @staticmethod
    def _parameter_indices(
        table: npt.NDArray[Any],
        references: Mapping[Expression, ReferenceSpec],
        owner: str,
    ) -> dict[Expression, int]:
        """Return the flat parameter index of every symbol of a table, after checking that it is a parameter."""
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

    @staticmethod
    def _checked_rate(law: KineticLaw, sizes: tuple[int, int, int], owner: str) -> npt.NDArray[Any]:
        """Return the rate table of a law, after checking that it has one row per candidate and one column per field."""
        rate = CompileEncountersPass._kinetic_table(law.rate)
        if rate.ndim and rate.shape != sizes[:2]:
            error_msg = (
                f'An encounter law of {owner} has a rate of shape {rate.shape}, but its candidate and field groups '
                f'need the shape {sizes[:2]}.'
            )
            raise ValidationError(error_msg)
        return rate

    @staticmethod
    def _outcome_table(law: KineticLaw, sizes: tuple[int, int, int], owner: str) -> npt.NDArray[Any] | None:
        """Return the outcome table of a law, the identity on the candidate's activity for a bare proliferation."""
        if isinstance(law, DestructionLaw):
            return None
        value = law.transition if isinstance(law, EncounterLaw) else law.outcomes
        if value is None:
            if sizes[0] != sizes[2]:
                error_msg = (
                    f'A proliferation of {owner} has no outcome table, so the new particles keep the candidate '
                    f'activity, but the test group has {sizes[2]} activities and the candidate {sizes[0]}. Give the '
                    'outcome table.'
                )
                raise ValidationError(error_msg)
            identity = np.zeros(sizes)
            identity[np.arange(sizes[0]), :, np.arange(sizes[0])] = 1.0
            return identity
        table = CompileEncountersPass._kinetic_table(value)
        if table.shape != sizes:
            error_msg = (
                f'An encounter law of {owner} has an outcome table of shape {table.shape}, but its candidate, field, '
                f'and test groups need the shape {sizes}.'
            )
            raise ValidationError(error_msg)
        return table

    @staticmethod
    def _compile_encounter(
        law: KineticLaw,
        owner: str,
        references: Mapping[Expression, ReferenceSpec],
        time: Expression,
        *,
        fused: bool = False,
    ) -> EncounterKernel:
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
        candidate = CompileEncountersPass._group_indices(law.candidate, references, owner)
        field_indices = CompileEncountersPass._group_indices(law.field, references, owner)
        test_symbol = getattr(law, 'test', None)
        test = (
            candidate if test_symbol is None else CompileEncountersPass._group_indices(test_symbol, references, owner)
        )
        sizes = (candidate.size, field_indices.size, test.size)
        rates = CompileEncountersPass._checked_rate(law, sizes, owner)
        table = CompileEncountersPass._outcome_table(law, sizes, owner)
        grid = tuple(item.ravel().astype(np.intp) for item in np.indices(sizes[:2]))
        rate = TableValuesSpec(rates, grid, CompileEncountersPass._parameter_indices(rates, references, owner), time)
        if table is None:
            empty = np.zeros(0, dtype=np.intp)
            return EncounterKernel(
                owner, CompileEncountersPass._effect(law), candidate, field_indices, test, rate, (empty,) * 3, None,
            ).check()
        positions = CompileEncountersPass._nonzero(table)
        entries = (positions[0], positions[1], positions[2])
        probabilities = TableValuesSpec(
            table, positions, CompileEncountersPass._parameter_indices(table, references, owner), time,
        )
        return EncounterKernel(
            owner,
            CompileEncountersPass._effect(law),
            candidate,
            field_indices,
            test,
            rate,
            entries,
            probabilities,
            fused,
        ).check()

    @staticmethod
    def _compile_encounters(
        components: Sequence[tuple[str, Mapping[Expression, ReferenceSpec], Sequence[BaseLaw]]],
        time: Expression,
        *,
        fused: bool = False,
    ) -> tuple[EncounterKernel, ...]:
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
            CompileEncountersPass._compile_encounter(law, owner, references, time, fused=fused)
            for owner, references, laws in components
            for law in laws
            if isinstance(law, _KINETIC_LAWS)
        )

    @staticmethod
    def _kinetic_table(value: object) -> npt.NDArray[Any]:
        """Return a rate or outcome table of an encounter law as an array of floats, or of Sage expressions.

        Args:
            value:
                The table, a number, a Sage expression, or a nested sequence of them.

        Returns:
            table:
                A float array when every entry is a number, otherwise an object array of Sage expressions.

        Examples:
            >>> from skcomplex.engine.compile.passes._compile_encounters import CompileEncountersPass
            >>> CompileEncountersPass._kinetic_table([[0.5, 0.5]]).dtype
            dtype('float64')
        """
        array = np.asarray(value, dtype=object)
        try:
            return np.asarray(array, dtype=float)
        except TypeError:
            return np.asarray(np.vectorize(SR, otypes=[object])(array), dtype=object)
