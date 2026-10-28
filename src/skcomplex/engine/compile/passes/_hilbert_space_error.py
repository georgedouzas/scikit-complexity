"""A quantum system has a Hilbert space larger than its declared limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....errors import SkcomplexError


class HilbertSpaceError(SkcomplexError):
    """A quantum system has a Hilbert space larger than its declared limit."""
