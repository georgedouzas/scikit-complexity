"""Define the errors of the gallery tooling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class GalleryError(Exception):
    """Error of the gallery: a failed capture, a failed check, or an example over its budget."""
