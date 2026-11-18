"""Define the spec of a reaction of a metabolic network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from .....core.errors import ValidationError
from .....core.specs import BaseSpec

DEFAULT_BOUND = 1000.0


@dataclass(frozen=True)
class ReactionSpec(BaseSpec):
    """Reaction of a metabolic network.

    It has its stoichiometry, its bounds, and the genes that gate them.

    Args:
        coefficients:
            The stoichiometric coefficient of every metabolite, negative for a reactant.

        lower:
            The lower bound of the flux.

        upper:
            The upper bound of the flux.

        rule:
            The gene rule, a Boolean expression of gene names with `and`, `or`, and parentheses, or the empty string for
            a reaction without genes.

    Raises:
        ValidationError:
            The lower bound is above the upper bound, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import ReactionSpec
        >>> ReactionSpec({'A': -1.0, 'B': 1.0}, rule='g1 or g2').upper
        1000.0
    """

    coefficients: Mapping[str, float]
    lower: float = 0.0
    upper: float = DEFAULT_BOUND
    rule: str = ''

    def _check(self) -> None:
        """Raise an error for a lower bound above the upper bound.

        Raises:
            ValidationError:
                The lower bound is above the upper bound.
        """
        if self.lower > self.upper:
            error_msg = f'The lower bound {self.lower} of a reaction is above its upper bound {self.upper}.'
            raise ValidationError(error_msg)
