"""Define the listing of a system: its name and metadata, and the module that registers it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemListing:
    """System as the catalog lists it: its name and metadata, and the module that registers it.

    Args:
        name:
            The name the system is created by.

        field:
            The field path of the system, such as `biology/ecology`.

        description:
            One sentence on what the system shows.

        tags:
            Keywords for filtering.

        references:
            The sources of the system.

        module:
            The dotted name of the module that registers the system.
    """

    name: str
    field: str
    description: str
    tags: tuple[str, ...]
    references: tuple[str, ...]
    module: str
