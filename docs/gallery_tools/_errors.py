"""Define the errors and the warnings of the gallery tooling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class GalleryError(Exception):
    """Error of the gallery.

    It is a failed capture, a failed check, or an example over its budget.
    """


class GalleryWarning(UserWarning):
    """Warning of the gallery.

    It warns about a thumbnail that may not show what its example describes.
    """
