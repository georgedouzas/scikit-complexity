"""The update of the lowest value of a search from an observable of another partition."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from ....core.base import BasePart
from ....core.interactions import Interaction
from ....core.laws import UpdateLaw
from ....core.laws.utils import where
from ....specs import RoleSpec


class RunningBestInteraction(Interaction):
    """Keep the lowest value of an observable, `best <- value if value < best else best`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds `best`.

        source:
            The path of the observable.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import RunningBestInteraction
        >>> RunningBestInteraction('update', target='record', source='search/lowest').source
        'search/lowest'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the update of the lowest value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of `best`.
        """
        return [UpdateLaw(s.target.best, where(s.source.value < s.target.best, s.source.value, s.target.best))]
