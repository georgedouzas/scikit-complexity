"""Substitution system of one patch.

It is compiled to batched products of affine matrices.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...components.elements import PatchElement
from ...components.laws import SubstitutionLaw
from ...errors import RewritingError
from ...specs import AffineSpec, BaseStructureSpec, ParametersSpec, PatchStructureSpec, RunStateSpec
from ._base import BaseKernel
from ._base_problem_spec import BaseProblemSpec
from ._word_kernel import IDENTIFIER


class PatchKernel(BaseKernel):
    """Substitution system of one patch.

    It is compiled to batched products of affine matrices.

    Args:
        path:
            The path of the patch.

        patch:
            The patch element.

        substitutions:
            The substitutions that target it.

        parameters:
            The parameter symbols of the system.

    Attributes:
        values_:
            The current tiles, set by `prepare` and rewritten by `run`.
    """

    values_: PatchStructureSpec

    def __init__(
        self,
        path: str,
        patch: PatchElement,
        substitutions: Sequence[SubstitutionLaw],
        parameters: ParametersSpec,
    ) -> None:
        self.path = path
        self.patch = patch
        self.substitutions = substitutions
        self.parameters = parameters

    def _compile(self) -> None:
        """Compile the prototiles, the tiles, and the substitutions.

        Every name is checked to be a prototile.

        Raises:
            RewritingError:
                The patch has no prototiles, a name is not a prototile, or a prototile has two substitutions.
        """
        path, patch, substitutions = self.path, self.patch, self.substitutions
        self.max_size = int(patch.max_size)
        self.prototiles = tuple(patch.prototiles)
        if not self.prototiles:
            error_msg = f'Patch {path!r} has no prototiles. Give at least one.'
            raise RewritingError(error_msg)
        for prototile in self.prototiles:
            prototile._check()
        self.names = tuple(prototile.name for prototile in self.prototiles)
        self.counted = tuple(name for name in dict.fromkeys(self.names) if IDENTIFIER.match(name))
        tiles = patch.tiles if patch.tiles is not None else [(self.names[0], None)]
        self.codes = np.array([self._get_code(name, 'the tiles') for name, _ in tiles], dtype=np.int64)
        self.placements = [placement for _, placement in tiles]
        self.pieces: dict[int, list[tuple[int, AffineSpec]]] = {}
        for substitution in substitutions:
            where = f'The substitution of {substitution.prototile!r}'
            code = self._get_code(substitution.prototile, where)
            if code in self.pieces:
                error_msg = f'Patch {path!r} has two substitutions of {substitution.prototile!r}. Keep one.'
                raise RewritingError(error_msg)
            self.pieces[code] = [(self._get_code(name, where), affine) for name, affine in substitution.pieces]

    def _get_code(self, name: str, where: str) -> int:
        """Return the code of a prototile by name."""
        if name not in self.names:
            error_msg = (
                f'{where} of patch {self.path!r} names the prototile {name!r}, which is not one of {self.names}.'
            )
            raise RewritingError(error_msg)
        return self.names.index(name)

    def _get_values(self, values: npt.NDArray[np.float64]) -> dict[Expression, float]:
        """Return the parameter values by symbol."""
        return dict(zip(self.parameters.symbols, values.tolist(), strict=True))

    def _prepare(self, context: object) -> None:
        """Start the structure at generation 0 for a run of the problem, as `values_`.

        It first compiles the structure and its laws, which checks them, so the compiler that reads the compiled tables
        before any run calls `_compile` alone.

        Args:
            context:
                The problem whose initial parameters the run starts from.
        """
        self._compile()
        if not isinstance(context, BaseProblemSpec):
            error_msg = f'Patch {self.path!r} starts from the initial parameters of a problem, but got none.'
            raise RewritingError(error_msg)
        self.values_ = self._get_initial(context.initial_parameters)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Rewrite the current structure once, in parallel.

        The rewrite reads the parameters of the state.
        """
        if inputs is None:
            error_msg = f'Patch {self.path!r} rewrites at a state, but got none.'
            raise RewritingError(error_msg)
        self.values_ = self._rewrite(self.values_, inputs.p)

    def _get_initial(self, values: npt.NDArray[np.float64]) -> PatchStructureSpec:
        """Return the patch at generation 0.

        Args:
            values:
                The parameter values.

        Returns:
            patch:
                The initial tiles.
        """
        by_symbol = self._get_values(values)
        matrices = [np.eye(3) if item is None else item.get_matrix_at(by_symbol) for item in self.placements]
        return PatchStructureSpec(
            self.prototiles,
            BaseStructureSpec._get_frozen(self.codes, np.int64),
            BaseStructureSpec._get_frozen(np.reshape(matrices, (-1, 3, 3)), np.float64),
            0,
        )

    def get_counts(self, patch: PatchStructureSpec) -> list[float]:
        """Return the count of every prototile and the number of tiles.

        Args:
            patch:
                The patch.

        Returns:
            counts:
                The counts, in the order of the states.
        """
        return [float(patch.count(name)) for name in self.counted] + [float(patch.parts)]

    def _rewrite(
        self,
        patch: PatchStructureSpec,
        values: npt.NDArray[np.float64],
    ) -> PatchStructureSpec:
        """Return the patch of the next generation.

        Every tile is replaced by its pieces in parallel.

        Args:
            patch:
                The patch.

            values:
                The parameter values.

        Returns:
            patch:
                The next patch.

        Raises:
            RewritingError:
                The next patch would exceed the maximum size.
        """
        by_symbol = self._get_values(values)
        codes, matrices = patch.codes, patch.matrices
        children = np.ones(len(self.prototiles), dtype=np.int64)
        for code, pieces in self.pieces.items():
            children[code] = len(pieces)
        lengths = children[codes]
        size = int(lengths.sum())
        if size > self.max_size:
            error_msg = (
                f'Patch {self.path!r} would reach {size} tiles at generation {patch.generation + 1}, above its maximum '
                f'size {self.max_size}. Raise max_size or run fewer generations.'
            )
            raise RewritingError(error_msg)
        offsets = np.cumsum(lengths) - lengths
        new_codes = np.empty(size, dtype=np.int64)
        new_matrices = np.empty((size, 3, 3))
        kept = np.flatnonzero(~np.isin(codes, list(self.pieces)))
        new_codes[offsets[kept]] = codes[kept]
        new_matrices[offsets[kept]] = matrices[kept]
        for code, pieces in self.pieces.items():
            rows = np.flatnonzero(codes == code)
            if not len(rows):
                continue
            maps = np.stack([affine.get_matrix_at(by_symbol) for _, affine in pieces])
            places = offsets[rows][:, None] + np.arange(len(pieces))
            new_matrices[places] = np.einsum('tij,pjk->tpik', matrices[rows], maps)
            new_codes[places] = [piece for piece, _ in pieces]
        return PatchStructureSpec(
            self.prototiles,
            BaseStructureSpec._get_frozen(new_codes, np.int64),
            BaseStructureSpec._get_frozen(new_matrices, np.float64),
            patch.generation + 1,
        )
