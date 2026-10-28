"""Signal that the dense blocks of a Jacobian pattern exceed the size limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class _TooDense(Exception):  # noqa: N818
    """Raised internally when the dense blocks of a pattern exceed the size limit."""
