"""Define the error for an operator system whose space of states is too large."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....errors import SkcomplexError


class HilbertSpaceError(SkcomplexError):
    """An error for an operator system whose Hilbert space exceeds its limit.

    The Hilbert space of a system of operators has one dimension per joint state of its elements, so it doubles with
    every qubit, and ten qubits already make 1024 dimensions. Use fewer elements or levels, or raise `max_dimension`
    when the memory allows. It bounds registers of qubits, chains of spins, and modes of light.

    Read more in the user guide, under Troubleshooting.

    See Also:
        OperatorCompiler: Compiles systems of operators.
        SpaceTooLargeError: Raised for a space with too many sites.

    Examples:
        >>> from skcomplex.core.engine.compile import HilbertSpaceError
        >>> from skcomplex.core.errors import SkcomplexError
        >>> issubclass(HilbertSpaceError, SkcomplexError)
        True
    """
