"""Exact Jacobian of a drift with its sparsity pattern."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, Protocol

from ...types import Array


class Jacobian(Protocol):
    """Exact Jacobian of a drift with its sparsity pattern."""

    def evaluate(self, t: float, x: Array, p: Array) -> Any:  # noqa: ANN401
        """Return the Jacobian at a state as a sparse matrix."""

    def pattern(self) -> Any:  # noqa: ANN401
        """Return the boolean structure of the Jacobian, or `None` when it is too dense to state."""

    def validate(self) -> None:
        """Raise an error when a law has no symbolic form whose exact Jacobian can be derived."""
