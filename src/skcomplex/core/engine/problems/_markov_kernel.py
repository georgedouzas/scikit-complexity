"""Transition or emission matrix of a Markov law.

It is evaluated from the parameters to check its rows.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ...specs import RunStateSpec
from ...types import Array
from ._base import BaseKernel

_ROW_TOLERANCE = 1e-9


class MarkovKernel(BaseKernel):
    """Transition or emission matrix of a Markov law.

    It is evaluated from the parameters to check its rows. `run(state)` keeps in `values_` the description of the first
    negative entry or row that does not sum to one at the parameters of the state, or `None` when every matrix is
    stochastic.

    Args:
        law:
            The description of the law, used in messages.

        function:
            The function `M(p)` that returns the matrix of every member from the parameter vector.

        reads:
            The indices of the parameter vector the matrix reads.

    Attributes:
        values_:
            The description of what is wrong with the matrix at the last run, or `None`.
    """

    values_: str | None

    def __init__(
        self,
        law: str,
        function: Callable[[Array], npt.NDArray[np.float64]],
        reads: frozenset[int],
    ) -> None:
        self.law = law
        self.function = function
        self.reads = reads

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the description of what is wrong with the matrix at the parameters.

        It is `None` when nothing is wrong.
        """
        self.values_ = self._problem(self._state_of(inputs))

    def _problem(self, state: RunStateSpec) -> str | None:
        """Return the description of what is wrong with the matrix at a state.

        The matrix is read at the parameters of the state, and `None` is returned when nothing is wrong.
        """
        values = self.function(state.p)
        for member, matrix in enumerate(values):
            where = f' of member {member}' if values.shape[0] > 1 else ''
            negative = np.argwhere(~(matrix >= 0))
            if negative.size:
                row, column = negative[0]
                return f'the entry ({row}, {column}){where} of the {self.law} is {matrix[row, column]}, below 0'
            sums = matrix.sum(axis=1)
            wrong = np.flatnonzero(np.abs(sums - 1) > _ROW_TOLERANCE)
            if wrong.size:
                return f'the row {wrong[0]}{where} of the {self.law} sums to {sums[wrong[0]]}, not 1'
        return None
