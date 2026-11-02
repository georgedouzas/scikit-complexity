"""Define the errors and the warnings of the gallery tooling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class GalleryError(Exception):
    """Error of the gallery: a failed capture, a failed check, or an example over its budget."""


class GalleryWarning(UserWarning):
    """Warning of the gallery: a thumbnail that may not show what its example describes."""
