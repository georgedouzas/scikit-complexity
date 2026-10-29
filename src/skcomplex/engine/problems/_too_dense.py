"""Signal that the dense blocks of a Jacobian pattern exceed the size limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class _TooDenseError(Exception):
    """Raised internally when the dense blocks of a pattern exceed the size limit."""
