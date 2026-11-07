"""Define the listing of a system: its name and metadata, and the module that registers or defines it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemListing:
    """System as the catalog lists it: its name and metadata, and the module that registers or defines it.

    A system defined as a class is listed under the name of its class without the suffix `System`, in snake case, so
    `GrayScottSystem` is listed as `gray_scott`, with its topics as its tags.

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
            The dotted name of the module that registers or defines the system.

        class_name:
            The name of the class of a system defined as a class, or `None` for a system registered as a factory.
    """

    name: str
    field: str
    description: str
    tags: tuple[str, ...]
    references: tuple[str, ...]
    module: str
    class_name: str | None = None
