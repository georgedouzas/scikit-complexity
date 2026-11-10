"""An operator system has a Hilbert space larger than its declared limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....errors import SkcomplexError


class HilbertSpaceError(SkcomplexError):
    """An error for an operator system whose Hilbert space exceeds its declared limit."""
