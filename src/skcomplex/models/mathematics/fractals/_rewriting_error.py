"""Define the error of malformed rewriting systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....errors import ValidationError


class RewritingError(ValidationError):
    """A rewriting system is malformed, or a generation would exceed its maximum size."""
