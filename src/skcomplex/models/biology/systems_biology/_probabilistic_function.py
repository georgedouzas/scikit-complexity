"""Synchronous map of a node by one of several truth tables, drawn with given probabilities at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from math import prod
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression

from ....core.base import BasePart
from ....core.interactions import MapInteraction
from ....core.laws import UpdateLaw
from ....core.laws.utils import categorical, where
from ....errors import ValidationError
from ....specs import RoleSpec


class ProbabilisticFunctionInteraction(MapInteraction):
    """Synchronous map of a node by one of several truth tables, drawn with given probabilities at every step.

    Each table of `2^k` entries reads the `k` inputs as the bits of a binary number, from the most significant, and
    gives the next value `x` of the target. A second update of the same step records the index of the drawn table in
    the state `function` of the target, from the same draw.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node, a `ProbabilisticNodeElement`.

        inputs:
            The paths of the input nodes, in the order of the bits of the tables.

        tables:
            The candidate truth tables.

        probabilities:
            The probability of every table.

    Raises:
        ValidationError:
            The tables and the probabilities differ in number, or a table does not have one entry per pattern.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ProbabilisticFunctionInteraction
        >>> function = ProbabilisticFunctionInteraction(
        ...     'f', target='c', inputs=['a', 'b'], tables=((0, 0, 0, 1), (0, 1, 1, 1)), probabilities=(0.7, 0.3)
        ... )
        >>> function.probabilities
        (0.7, 0.3)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        tables: Sequence[Sequence[int]] = ((0,),),
        probabilities: Sequence[float] = (1.0,),
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._get_drawn_value)
        self.inputs = inputs
        self.tables = tables
        self.probabilities = probabilities

    def _get_pattern(self, inputs: Sequence[Expression], index: int) -> Expression:
        """Return the product that is 1 when the inputs, read as bits from the most significant, spell an index."""
        bits = [(index >> (len(inputs) - 1 - position)) & 1 for position in range(len(inputs))]
        return prod((value if bit else 1 - value for value, bit in zip(inputs, bits, strict=True)), start=SR(1))

    def _get_table_value(self, inputs: list[Expression], table: Sequence[int]) -> Expression:
        """Return the value of one truth table of the inputs."""
        return sum((self._get_pattern(inputs, index) for index, entry in enumerate(table) if entry), start=SR(0))

    def _check(self) -> None:
        """Raise an error for tables and probabilities that differ in number, or a table of the wrong size.

        Raises:
            ValidationError:
                The tables and the probabilities differ in number, or a table does not have one entry per pattern.
        """
        super()._check()
        if len(self.tables) != len(self.probabilities):
            error_msg = (
                f'ProbabilisticFunction {self.label} has {len(self.tables)} tables and {len(self.probabilities)} '
                'probabilities. Give one probability per table.'
            )
            raise ValidationError(error_msg)
        patterns = 2 ** len(self.inputs)
        if any(len(table) != patterns for table in self.tables):
            error_msg = (
                f'ProbabilisticFunction {self.label} reads {len(self.inputs)} inputs, which take {patterns} '
                'patterns, but one of its tables has another number of entries.'
            )
            raise ValidationError(error_msg)

    def _get_draw(self) -> Expression:
        """Return the index of the table drawn in a step, one draw for the value and its record."""
        return categorical('function', list(self.probabilities))

    def _get_drawn_value(self, namespace: SimpleNamespace) -> Expression:
        """Return the value of the drawn truth table of the inputs."""
        inputs = [item.x for item in namespace.inputs]
        chosen = self._get_draw()
        values = [self._get_table_value(inputs, table) for table in self.tables]
        value = values[-1]
        for index in range(len(values) - 2, -1, -1):
            value = where(chosen == index, values[index], value)
        return value

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the map of the target by the drawn table and the record of the drawn index.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of the value and the update of the index of the drawn table.
        """
        return [*super()._get_laws(namespace), UpdateLaw(namespace.target.function, self._get_draw())]
