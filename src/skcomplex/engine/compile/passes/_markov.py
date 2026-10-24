"""Take the Markov laws out of the laws of every component and check their matrices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

import numpy as np

from ....core.base import ValidationError
from .._compilation import BlockLaws, Compilation
from .._declarations import Declarations
from .._pipeline import Pass
from .._problem import _MarkovMatrix
from .._specs import KernelSpec
from .._stochastic import MatrixLaw, compile_matrix, markov_laws


def _checked_matrices(
    markov: list[tuple[KernelSpec, MatrixLaw]],
    declarations: Declarations,
) -> tuple[_MarkovMatrix, ...]:
    """Return the matrices of the Markov laws after checking them at the initial parameter values."""
    initial = np.concatenate([np.zeros(0), *declarations.initial_parameters])
    matrices = []
    for spec, law in markov:
        matrix = compile_matrix(law, spec.references, max(spec.rows, 1), declarations.time)
        violation = matrix.violation(initial)
        if violation is not None:
            error_msg = (
                f'At the initial parameter values, {violation}. Give a matrix with nonnegative entries whose rows sum '
                'to 1.'
            )
            raise ValidationError(error_msg)
        matrices.append(matrix)
    return tuple(matrices)


class MarkovPass(Pass):
    """Pass that rewrites transition and emission laws into Markov matrices checked at the initial values."""

    reads = ('lowered', 'layout')
    writes = ('lowered', 'markov', 'matrices')

    def run(self, compilation: Compilation) -> None:
        """Take the Markov laws out of the lowered laws.

        Args:
            compilation:
                The compilation state.
        """
        lowered: BlockLaws = []
        markov: list[tuple[KernelSpec, MatrixLaw]] = []
        for spec, laws, block in cast('BlockLaws', compilation.lowered):
            kept, matrices = markov_laws(spec.owner, laws, spec.references)
            lowered.append((spec, kept, block))
            markov.extend((spec, matrix) for matrix in matrices)
        compilation.lowered = lowered
        compilation.markov = markov
        compilation.matrices = _checked_matrices(markov, cast('Declarations', compilation.layout))
