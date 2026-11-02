"""Synchronous update of a node by a truth table of any number of input nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from math import prod
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression

from ....core.interactions import MapInteraction
from ....errors import ValidationError
from ....specs import RoleSpec


class BooleanFunctionInteraction(MapInteraction):
    """Synchronous update of a node by a truth table of any number of input nodes.

    A table of `2^k` entries reads the `k` inputs as the bits of a binary number, from the most significant, and gives
    the next value of the target. The update is the sum, over the patterns whose entry is 1, of the product that is 1
    exactly on that pattern.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node.

        inputs:
            The paths of the input nodes, in the order of the bits of the table.

        table:
            The next value of the target for each of the `2^k` input patterns.

    Raises:
        ValidationError:
            The table does not have one entry per pattern of the inputs.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import BooleanFunctionInteraction
        >>> BooleanFunctionInteraction('and', target='c', inputs=['a', 'b'], table=(0, 0, 0, 1)).table
        (0, 0, 0, 1)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        table: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._get_next)
        self.inputs = inputs
        self.table = table

    def _get_pattern(self, inputs: Sequence[Expression], index: int) -> Expression:
        """Return the product that is 1 when the inputs, read as bits from the most significant, spell an index."""
        bits = [(index >> (len(inputs) - 1 - position)) & 1 for position in range(len(inputs))]
        return prod((value if bit else 1 - value for value, bit in zip(inputs, bits, strict=True)), start=SR(1))

    def _get_next(self, s: SimpleNamespace) -> Expression:
        """Return the next value of the target by the truth table."""
        inputs = [item.x for item in s.inputs]
        terms = [self._get_pattern(inputs, index) for index, entry in enumerate(self.table) if entry]
        return sum(terms, start=SR(0))

    def _check(self) -> None:
        """Raise an error for a table without one entry per pattern of the inputs.

        Raises:
            ValidationError:
                The table does not have one entry per pattern of the inputs.
        """
        super()._check()
        if len(self.table) != 2 ** len(self.inputs):
            error_msg = (
                f'BooleanFunction {self.label} reads {len(self.inputs)} inputs, which take {2 ** len(self.inputs)} '
                f'patterns, but its table has {len(self.table)} entries.'
            )
            raise ValidationError(error_msg)
