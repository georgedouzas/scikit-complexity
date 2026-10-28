"""Define the matrix law spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from ._base_spec import BaseSpec

_RANDOM_FIELDS = ('value', 'rate', 'changes', 'lhs', 'rhs', 'guard', 'reset', 'waiting')


@dataclass(frozen=True)
class MatrixLawSpec(BaseSpec):
    """Matrix of a Markov law of a component, with the description of the law.

    Args:
        owner:
            The path of the component.

        description:
            The description of the law.

        matrix:
            The rows of the matrix.
    """

    owner: str
    description: str
    matrix: Sequence[Sequence[object]]
