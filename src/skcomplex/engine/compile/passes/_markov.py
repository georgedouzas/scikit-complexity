"""Take the Markov laws out of the laws of every component and check their matrices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ....core.laws import BaseLaw, EmissionLaw, TransitionLaw, UpdateLaw
from ....core.laws.utils import categorical, where
from ....errors import ValidationError
from ....specs import CallSpec, EntrySpec, MatrixLawSpec, RunStateSpec
from ....types import Array, DomainKind
from ..._utils import symbols_of
from ...problems import DeclarationsSpec, ExpressionPrinter, KernelSpec, ReferenceSpec
from ...problems._markov_kernel import MarkovKernel
from ...types import ReadKind
from .._base import BasePass
from .._compilation import BlockLaws, Compilation

SHARED_NDIM = 2
MEMBER_NDIM = 3


class MarkovPass(BasePass):
    """Pass that rewrites transition and emission laws into Markov matrices checked at the initial values."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered', 'markov', 'matrices')

    def _run(self, inputs: object = None) -> None:
        """Take the Markov laws out of the lowered laws."""
        compilation = self.compilation_
        lowered: BlockLaws = []
        markov: list[tuple[KernelSpec, MatrixLawSpec]] = []
        for spec, laws, block in Compilation._required(compilation.lowered, 'lowered'):
            kept, matrices = self._markov_laws(spec, laws)
            lowered.append((spec, kept, block))
            markov.extend((spec, matrix) for matrix in matrices)
        compilation.lowered = lowered
        compilation.markov = markov
        compilation.matrices = self._checked_matrices(markov, Compilation._required(compilation.layout, 'layout'))

    @staticmethod
    def _checked_matrices(
        markov: list[tuple[KernelSpec, MatrixLawSpec]],
        declarations: DeclarationsSpec,
    ) -> tuple[MarkovKernel, ...]:
        """Return the matrices of the Markov laws after checking them at the initial parameter values."""
        initial = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        matrices: list[MarkovKernel] = []
        for spec, law in markov:
            matrix = MarkovPass._compile_matrix(law, spec.references, max(spec.rows, 1), declarations.time)
            violation = matrix.run(RunStateSpec(0.0, np.zeros(0), initial)).values_
            if violation is not None:
                error_msg = (
                    f'At the initial parameter values, {violation}. Give a matrix with nonnegative entries whose rows '
                    'sum to 1.'
                )
                raise ValidationError(error_msg)
            matrices.append(matrix)
        return tuple(matrices)

    @staticmethod
    def _alphabet_size(description: str, reference: ReferenceSpec | None) -> int | None:
        """Return the number of values of a state over a finite alphabet, or `None` for the integers."""
        if reference is None or not reference.is_state:
            error_msg = f'The {description} must target a scalar state of its component.'
            raise ValidationError(error_msg)
        domain = reference.entry.domain
        labels = EntrySpec._labels_of(domain)
        if labels is not None:
            return len(labels)
        if EntrySpec._domain_kind(domain) == DomainKind.BOOLEAN:
            return 2
        if EntrySpec._domain_kind(domain) == DomainKind.INTEGER:
            return None
        error_msg = (
            f'The {description} moves {reference.entry.name}, whose domain {domain} is not a finite alphabet. Declare '
            'the state with a categorical or an integer domain.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _check_shape(
        description: str,
        matrix: Sequence[Sequence[object]],
        rows: int | None,
        columns: int | None,
    ) -> None:
        """Raise an error when a matrix does not have the rows and columns its alphabets need."""
        shape = (len(matrix), len(matrix[0]))
        if (rows is None or shape[0] == rows) and (columns is None or shape[1] == columns):
            return
        needed = (
            f'{rows if rows is not None else shape[0]} rows and {columns if columns is not None else shape[1]} columns'
        )
        error_msg = (
            f'The matrix of the {description} has {shape[0]} rows and {shape[1]} columns, but it needs {needed}, one '
            'row per value of the selecting state and one column per value of the target.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _draw_name(kind: str, target: Expression) -> str:
        """Return the draw name of a Markov law, which no user draw name starts with."""
        return f'_{kind}_{re.sub(r"[^0-9A-Za-z]", "_", str(target))}'

    @staticmethod
    def _matrix_reader(reference: ReferenceSpec) -> Callable[[Array], Array]:
        """Return the function that reads a parameter of a matrix for every evaluation row."""
        if reference.source is ReadKind.MEMBER:
            members = reference.members
            return lambda p: p[members]
        index = reference.index
        return lambda p: p[index]

    @staticmethod
    def _selected_row(selector: Expression, matrix: Sequence[Sequence[object]]) -> list[Expression]:
        """Return the row of a matrix that the value of a selecting expression picks, as nested conditions."""
        weights = []
        for column in range(len(matrix[0])):
            weight = SR(matrix[-1][column])
            for row in range(len(matrix) - 2, -1, -1):
                weight = where(selector == row, SR(matrix[row][column]), weight)
            weights.append(weight)
        return weights

    @staticmethod
    def _compile_matrix(
        law: MatrixLawSpec,
        references: Mapping[Expression, ReferenceSpec],
        rows: int,
        time: Expression,
    ) -> MarkovKernel:
        """Return the matrix of a Markov law as a function of the parameter vector.

        Args:
            law:
                The matrix of the law.

            references:
                The references of the symbols of the component.

            rows:
                The number of evaluation rows, the members of a group.

            time:
                The time symbol.

        Returns:
            matrix:
                The matrix as a function of the parameter vector.

        Raises:
            ValidationError:
                An entry of the matrix reads a state, time, or a symbol of another component.
        """
        entries = [SR(entry) for row in law.matrix for entry in row]
        arguments = sorted(symbols_of(entries), key=str)
        for item in arguments:
            reference = references.get(item)
            if reference is None or reference.is_state:
                error_msg = (
                    f'The matrix of the {law.description} reads {item}, but its entries must be numbers or expressions '
                    'of the parameters of its component.'
                )
                raise ValidationError(error_msg)
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run(entries).code_
        if not callable(function):
            error_msg = f'The printer returned a {type(function).__name__}, not the function of the matrix entries.'
            raise TypeError(error_msg)
        readers = [MarkovPass._matrix_reader(references[item]) for item in arguments]
        shape = (len(law.matrix), len(law.matrix[0]))
        reads = frozenset(
            int(index)
            for item in arguments
            for index in (
                references[item].flat_indices
                if references[item].source is ReadKind.MEMBER
                else [references[item].index]
            )
        )

        def evaluate(p: Array) -> npt.NDArray[np.float64]:
            values = function(0.0, *(reader(p) for reader in readers))
            if not isinstance(values, list):
                error_msg = f'The function of the matrix entries returned a {type(values).__name__}, not a list.'
                raise TypeError(error_msg)
            stacked = np.stack([np.broadcast_to(np.asarray(value, dtype=float), (rows,)) for value in values], axis=-1)
            return stacked.reshape(rows, *shape)

        return MarkovKernel(law.description, evaluate, reads).prepare()

    @staticmethod
    def _emission(
        owner: str,
        law: EmissionLaw,
        references: Mapping[Expression, ReferenceSpec],
        updates: dict[str, object],
    ) -> tuple[UpdateLaw, MatrixLawSpec]:
        """Return the update law an emission law becomes, reading the hidden state's next value, and its matrix."""
        description = f'emission law of {owner}.{law._get_targets()[0]}'
        columns = MarkovPass._alphabet_size(description, references.get(law._get_targets()[0]))
        rows = MarkovPass._alphabet_size(description, references.get(law.source))
        MarkovPass._check_shape(description, law.matrix, rows, columns)
        hidden = updates.get(str(law.source))
        if hidden is None:
            error_msg = (
                f'The {description} reads the next value of {law.source}, which no update or transition law of {owner} '
                'sets. Give the hidden state a transition law in the same component.'
            )
            raise ValidationError(error_msg)
        value = categorical(
            MarkovPass._draw_name('emission', law._get_targets()[0]),
            MarkovPass._selected_row(SR(hidden), law.matrix),
        )
        return UpdateLaw(law._get_targets()[0], value), MatrixLawSpec(owner, description, law.matrix)

    @staticmethod
    def _transition(spec: KernelSpec, law: TransitionLaw) -> tuple[UpdateLaw, MatrixLawSpec | None]:
        """Return the update law a transition law becomes, and its matrix unless a function gives it."""
        owner, references = spec.owner, spec.references
        if isinstance(law.matrix, CallSpec) or callable(law.matrix):
            return MarkovPass._function_transition(spec, law, law.matrix), None
        description = f'transition law of {owner}.{law._get_targets()[0]}'
        size = MarkovPass._alphabet_size(description, references.get(law._get_targets()[0]))
        MarkovPass._check_shape(description, law.matrix, size, size)
        MarkovPass._check_shape(description, law.matrix, len(law.matrix), len(law.matrix))
        value = categorical(
            MarkovPass._draw_name('transition', law._get_targets()[0]),
            MarkovPass._selected_row(law._get_targets()[0], law.matrix),
        )
        return UpdateLaw(law._get_targets()[0], value), MatrixLawSpec(owner, description, law.matrix)

    @staticmethod
    def _next_values(rows: Array, draws: Array, description: str) -> npt.NDArray[np.intp]:
        """Return the next value of every member from the row of its matrix, after checking the rows."""
        if np.any(rows < 0) or not np.allclose(rows.sum(axis=1), 1.0):
            error_msg = (
                f'The {description} returned {rows}. Give a matrix with nonnegative entries whose rows sum to 1.'
            )
            raise ValidationError(error_msg)
        return np.minimum((draws[:, None] >= np.cumsum(rows, axis=1)).sum(axis=1), rows.shape[1] - 1)

    @staticmethod
    def _function_transition(
        spec: KernelSpec,
        law: TransitionLaw,
        given: CallSpec | Callable[..., object],
    ) -> UpdateLaw:
        """Return the update law of a transition whose matrix is a function, which draws from the random stream."""
        target = law._get_targets()[0]
        reference = spec.references.get(target)
        description = f'transition law of {spec.owner}.{target}'
        size = MarkovPass._alphabet_size(description, reference)
        call = given if isinstance(given, CallSpec) else CallSpec(given, None)
        function = call.function
        if call.arguments or call.random or call.vectorized or not callable(function) or reference is None:
            error_msg = (
                f'The {description} has the matrix {given!r}. Give a function `f(t, values)` or '
                '`CallSpec(f, relative=True)`.'
            )
            raise ValidationError(error_msg)
        names = spec._relative_names([reference.entry.name])
        name = next(iter(names)) if call.relative else reference.entry.name

        def step(t: float, values: Mapping[str, object], *, rng: np.random.Generator) -> npt.NDArray[np.intp]:
            current = np.atleast_1d(np.asarray(values[name], dtype=float)).astype(np.intp)
            matrix = np.asarray(function(t, values), dtype=float)
            if matrix.ndim not in {SHARED_NDIM, MEMBER_NDIM} or (
                size is not None and matrix.shape[-2:] != (size, size)
            ):
                error_msg = f'The {description} returned a matrix of shape {matrix.shape}, not one row per value.'
                raise ValidationError(error_msg)
            rows = matrix[current] if matrix.ndim == SHARED_NDIM else matrix[np.arange(current.size), current]
            return MarkovPass._next_values(rows, rng.random(current.size), description)

        return UpdateLaw(target, replace(call, function=step, random=True))

    @staticmethod
    def _markov_laws(
        spec: KernelSpec,
        laws: list[BaseLaw],
    ) -> tuple[list[BaseLaw], list[MatrixLawSpec]]:
        """Return the laws of a component with transition and emission laws rewritten as update laws with draws.

        Args:
            spec:
                The kernel spec of the component, with its path and the references of the symbols the laws read.

            laws:
                The laws of the component.

        Returns:
            rewritten:
                The laws with every Markov law replaced by its update law, in the same order, and the matrices.

        Raises:
            ValidationError:
                A Markov law targets a state that is not over a finite alphabet, its matrix has the wrong shape, or an
                emission reads a hidden state that no update law of the component sets.
        """
        owner, references = spec.owner, spec.references
        matrices: list[MatrixLawSpec] = []
        rewritten: list[BaseLaw] = []
        for law in laws:
            if isinstance(law, TransitionLaw):
                update, found = MarkovPass._transition(spec, law)
                rewritten.append(update)
                matrices.extend([found] if found is not None else [])
            else:
                rewritten.append(law)
        updates = {str(law._get_targets()[0]): law.value for law in rewritten if isinstance(law, UpdateLaw)}
        for index, law in enumerate(rewritten):
            if isinstance(law, EmissionLaw):
                rewritten[index], matrix = MarkovPass._emission(owner, law, references, updates)
                matrices.append(matrix)
        return rewritten, matrices
