"""Define the local operator spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import qutip
import scipy.sparse as sp

from ._base_spec import BaseSpec

ADJOINT_NAMES = {'plus': 'minus', 'minus': 'plus', 'annihilate': 'create', 'create': 'annihilate'}
ADJOINT_SUFFIX = '.dag'


@dataclass(frozen=True)
class LocalOperatorSpec(BaseSpec):
    """Operator that acts on the Hilbert space of one subsystem.

    It is identified by its name.

    Args:
        name:
            The name of the operator, unique among the operators of its subsystem.

        matrix:
            The matrix of the operator.

        hermitian:
            Whether the operator equals its adjoint.

    Examples:
        >>> import qutip
        >>> from skcomplex.core.specs import LocalOperatorSpec
        >>> LocalOperatorSpec('plus', qutip.sigmap()).get_adjoint().name
        'minus'
    """

    name: str
    matrix: object = field(default=None, compare=False, repr=False)
    hermitian: bool = field(default=False, compare=False)

    def get_adjoint(self) -> 'LocalOperatorSpec':
        """Return the adjoint of the operator.

        Returns:
            adjoint:
                The operator itself when it is Hermitian, or the operator with the conjugate transposed matrix.
        """
        if self.hermitian:
            return self
        if self.name in ADJOINT_NAMES:
            name = ADJOINT_NAMES[self.name]
        elif self.name.endswith(ADJOINT_SUFFIX):
            name = self.name.removesuffix(ADJOINT_SUFFIX)
        else:
            name = f'{self.name}{ADJOINT_SUFFIX}'
        matrix = self._get_sparse_matrix(self.matrix).conj().T.tocsr()
        return LocalOperatorSpec(name, matrix, hermitian=False)

    @staticmethod
    def _get_sparse_matrix(matrix: object) -> sp.csr_array:
        """Return a matrix, a QuTiP object, or an array as a complex sparse array."""
        if isinstance(matrix, qutip.Qobj):
            dense = matrix.full()
        elif isinstance(matrix, sp.sparray | sp.spmatrix):
            dense = matrix.toarray()
        else:
            dense = matrix
        return sp.csr_array(np.asarray(dense, dtype=complex))
