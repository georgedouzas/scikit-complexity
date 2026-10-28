"""Define the listing of a model: its name and metadata, and the module that registers it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelListing:
    """Model as the catalog lists it: its name and metadata, and the module that registers it.

    Args:
        name:
            The name the model is created by.

        field:
            The field path of the model, such as `biology/ecology`.

        description:
            One sentence on what the model shows.

        tags:
            Keywords for filtering.

        references:
            The sources of the model.

        module:
            The dotted name of the module that registers the model.
    """

    name: str
    field: str
    description: str
    tags: tuple[str, ...]
    references: tuple[str, ...]
    module: str
