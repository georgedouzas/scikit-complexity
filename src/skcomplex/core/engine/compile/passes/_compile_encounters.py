"""Pass that compiles and takes out the encounter laws.

It runs before reductions prune the groups the laws read.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ....components.elements import FieldElement
from ....components.laws import BaseLaw, BilinearLaw
from ....components.laws.utils import coordinate
from ....components.laws.utils._phase import COORDINATE_PREFIX
from ....components.spaces import PhaseSpace
from ....errors import ValidationError
from ....specs import TableValuesSpec
from ....types import Indices
from ..._utils import symbols_of
from ...backends import BaseBackend
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._encounter_kernel import EncounterKernel
from ...problems._reference_spec import ReferenceSpec
from ...types import ReadKind
from .._base import BaseCompiler, BasePass
from .._compilation import Compilation, Laws


class CompileEncountersPass(BasePass):
    """Pass that compiles and takes out the encounter laws.

    It runs before reductions prune the groups the laws read.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn', 'encounters')

    def _run(self, inputs: object = None) -> None:
        """Compile the encounter laws of the drawn laws."""
        compilation = self.compilation_
        specs = Compilation._get_required(compilation.drawn, 'drawn')
        declarations = Compilation._get_required(compilation.layout, 'layout')
        compilation.encounters = self._get_compiled_encounters(specs, declarations, compilation.backend)
        compilation.drawn = [(spec, [law for law in laws if not isinstance(law, BilinearLaw)]) for spec, laws in specs]

    @staticmethod
    def _get_compiled_encounters(
        specs: Laws,
        declarations: DeclarationsSpec,
        backend: BaseBackend | None,
    ) -> tuple[EncounterKernel, ...]:
        """Return the encounter laws of every component, lowered for their members."""
        if not any(isinstance(law, BilinearLaw) for _, laws in specs for law in laws):
            return ()
        members = declarations.members
        fused = BaseCompiler._resolve_backend(backend, members)._get_name() == 'numba'
        components = [(spec.owner, dict(spec.references), laws) for spec, laws in specs]
        phases = {
            entry.name: block.node.component.space
            for block in declarations.blocks.values()
            if isinstance(block.node.component, FieldElement) and isinstance(block.node.component.space, PhaseSpace)
            for entry in block.entries.values()
        }
        return CompileEncountersPass._compile_encounters(components, declarations.time, fused=fused, phases=phases)

    @staticmethod
    def _get_nonzero(table: npt.NDArray[np.generic]) -> tuple[Indices, ...]:
        """Return the indices of the nonzero entries of a table.

        The entries are numbers or Sage expressions.
        """
        if table.dtype != object:
            return tuple(np.asarray(item, dtype=np.intp) for item in np.nonzero(table))
        mask = np.vectorize(lambda entry: not (SR(entry).is_numeric() and SR(entry) == 0), otypes=[bool])(table)
        return tuple(np.asarray(item, dtype=np.intp) for item in np.nonzero(mask))

    @staticmethod
    def _get_group_indices(
        item: Expression | None,
        references: Mapping[Expression, ReferenceSpec],
        owner: str,
    ) -> Indices:
        """Return the flat state indices of the density an encounter symbol stands for.

        There is one index per activity.
        """
        reference = references.get(item) if item is not None else None
        if reference is None or not reference.is_state or reference.source is not ReadKind.MEMBER:
            error_msg = (
                f'An encounter law of {owner} reads {item}, which is not a state of a group bound to a role. '
                'Encounters read the densities of groups on the sites of activity lattices, such as `s.candidate.x`.'
            )
            raise ValidationError(error_msg)
        return reference.flat_indices.astype(np.intp)

    @staticmethod
    def _get_parameter_indices(
        table: npt.NDArray[np.generic],
        references: Mapping[Expression, ReferenceSpec],
        owner: str,
    ) -> dict[Expression, int]:
        """Return the flat parameter index of every symbol of a table.

        It first checks that each symbol is a parameter.
        """
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
    def _get_rate_of_classes(rate: object, space: PhaseSpace | None, sizes: tuple[int, int, int]) -> object:
        """Return a rate that reads class coordinates as its table `eta[h, k]`.

        It is taken at the classes of the candidate.
        """
        names = [str(item) for item in SR(rate).variables()] if isinstance(rate, Expression) else []
        coordinates = [item for item in names if item.startswith(COORDINATE_PREFIX)]
        if space is None or not coordinates:
            return rate
        points = np.meshgrid(*(space.coordinates_[axis.name] for axis in space.axes), indexing='ij')
        grid = {coordinate(axis.name): point.ravel() for axis, point in zip(space.axes, points, strict=True)}
        rows = [SR(rate).subs({item: float(values[row]) for item, values in grid.items()}) for row in range(sizes[0])]
        return np.tile(np.asarray(rows, dtype=object)[:, None], (1, sizes[1]))

    @staticmethod
    def _get_checked_rate(
        law: BilinearLaw,
        sizes: tuple[int, int, int],
        owner: str,
        space: PhaseSpace | None = None,
    ) -> npt.NDArray[np.generic]:
        """Return the rate table of a law after checking its shape.

        It must have one row per candidate and one column per field.
        """
        rate = CompileEncountersPass._get_kinetic_table(
            CompileEncountersPass._get_rate_of_classes(law.rate, space, sizes),
        )
        if rate.ndim and rate.shape != sizes[:2]:
            error_msg = (
                f'An encounter law of {owner} has a rate of shape {rate.shape}, but its candidate and field groups '
                f'need the shape {sizes[:2]}.'
            )
            raise ValidationError(error_msg)
        return rate

    @staticmethod
    def _get_outcome_table(law: BilinearLaw, sizes: tuple[int, int, int], owner: str) -> npt.NDArray[np.generic]:
        """Return the outcome table of a law.

        Without one, it is the identity on the activity of the candidate.
        """
        value = law.outcomes
        if value is None:
            if sizes[0] != sizes[2]:
                error_msg = (
                    f'A bilinear law of {owner} has no outcome table, so the particles keep the candidate '
                    f'activity, but the target has {sizes[2]} activities and the candidate {sizes[0]}. Give the '
                    'outcome table.'
                )
                raise ValidationError(error_msg)
            identity = np.zeros(sizes)
            identity[np.arange(sizes[0]), :, np.arange(sizes[0])] = 1.0
            return identity
        table = CompileEncountersPass._get_kinetic_table(value)
        if table.shape != sizes:
            error_msg = (
                f'An encounter law of {owner} has an outcome table of shape {table.shape}, but its candidate, field, '
                f'and test groups need the shape {sizes}.'
            )
            raise ValidationError(error_msg)
        return table

    @staticmethod
    def _compile_encounter(
        law: BilinearLaw,
        owner: str,
        references: Mapping[Expression, ReferenceSpec],
        time: Expression,
        *,
        fused: bool = False,
        phases: Mapping[str, PhaseSpace] | None = None,
    ) -> EncounterKernel:
        """Return an encounter law lowered to sparse contractions over densities.

        The densities are those its roles bind.

        Args:
            law:
                The bilinear law.

            owner:
                The path of the component that returns the law.

            references:
                The references of the symbols of the component's namespace.

            time:
                The time symbol.

            fused:
                Whether the gain runs in one fused loop compiled with Numba.

            phases:
                The phase space of every entry of a field on one, whose encounters run at every position.

        Returns:
            encounter:
                The compiled law.

        Raises:
            ValidationError:
                A density is not a state of a group, a table has the wrong shape, or a table reads a state.
        """
        first, second = law.first, law.second
        candidate = CompileEncountersPass._get_group_indices(first, references, owner)
        linear = second is None
        empty = np.zeros(0, dtype=np.intp)
        field_indices = empty if linear else CompileEncountersPass._get_group_indices(second, references, owner)
        space = (phases or {}).get(references[first].entry.name)
        positions = 1 if space is None else space.position_size_
        test = CompileEncountersPass._get_group_indices(law.target, references, owner)
        sizes = (candidate.size // positions, max(field_indices.size // positions, 1), test.size // positions)
        rates = CompileEncountersPass._get_checked_rate(law, sizes, owner, space)
        table = CompileEncountersPass._get_outcome_table(law, sizes, owner)
        grid = tuple(item.ravel().astype(np.intp) for item in np.indices(sizes[:2]))
        rate = TableValuesSpec(
            rates,
            grid,
            CompileEncountersPass._get_parameter_indices(rates, references, owner),
            time,
        )
        nonzero = CompileEncountersPass._get_nonzero(table)
        entries = (nonzero[0], nonzero[1], nonzero[2])
        probabilities = TableValuesSpec(
            table,
            nonzero,
            CompileEncountersPass._get_parameter_indices(table, references, owner),
            time,
        )
        return EncounterKernel(
            owner,
            'transition' if law.conserved else 'bilinear',
            candidate,
            field_indices,
            test,
            rate=rate,
            entries=entries,
            probabilities=probabilities,
            fused=fused,
            positions=positions,
        ).prepare()

    @staticmethod
    def _compile_encounters(
        components: Sequence[tuple[str, Mapping[Expression, ReferenceSpec], Sequence[BaseLaw]]],
        time: Expression,
        *,
        fused: bool = False,
        phases: Mapping[str, PhaseSpace] | None = None,
    ) -> tuple[EncounterKernel, ...]:
        """Return the encounter laws of every component, lowered in declaration order.

        Args:
            components:
                The path, the references of the namespace, and the laws of every component.

            time:
                The time symbol.

            fused:
                Whether the gains run in fused loops compiled with Numba.

            phases:
                The phase space of every entry of a field on one.

        Returns:
            encounters:
                The compiled laws.
        """
        return tuple(
            CompileEncountersPass._compile_encounter(law, owner, references, time, fused=fused, phases=phases)
            for owner, references, laws in components
            for law in laws
            if isinstance(law, BilinearLaw)
        )

    @staticmethod
    def _get_kinetic_table(value: object) -> npt.NDArray[np.generic]:
        """Return a rate or outcome table of an encounter law as an array.

        The array holds floats or Sage expressions.

        Args:
            value:
                The table, a number, a Sage expression, or a nested sequence of them.

        Returns:
            table:
                A float array when every entry is a number, otherwise an object array of Sage expressions.

        Examples:
            >>> from skcomplex.core.engine.compile.passes._compile_encounters import CompileEncountersPass
            >>> CompileEncountersPass._get_kinetic_table([[0.5, 0.5]]).dtype
            dtype('float64')
        """
        array = np.asarray(value, dtype=object)
        try:
            return np.asarray(array, dtype=float)
        except TypeError:
            return np.asarray(np.vectorize(SR, otypes=[object])(array), dtype=object)
