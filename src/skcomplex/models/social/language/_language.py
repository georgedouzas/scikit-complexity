"""A language, which holds the exponent of its word frequencies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec


class LanguageElement(Element):
    """Language with the exponent `lambda` of its word frequencies.

    Args:
        label:
            The label of the language.

        exponent:
            The initial guess of the exponent, from which it is solved.

    Examples:
        >>> from skcomplex.models.social.language import LanguageElement
        >>> LanguageElement('language').exponent
        1.0
    """

    def __init__(self, label: str, exponent: Annotated[float, StateSpec()] = 1.0) -> None:
        super().__init__(label)
        self.exponent = exponent
