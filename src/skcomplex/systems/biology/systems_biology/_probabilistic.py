"""Probabilistic Boolean networks.

They draw the function of every node at every step.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from math import prod
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, SR, Expression
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.base import BasePart
from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import categorical, where
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from .types import Candidates

SHMULEVICH_INPUTS = ('x1', 'x2', 'x3')
SHMULEVICH_NETWORK: dict[str, tuple[tuple[tuple[int, ...], float], ...]] = {
    'x1': (((0, 1, 1, 1, 0, 1, 1, 1), 0.6), ((0, 0, 0, 1, 0, 1, 1, 1), 0.4)),
    'x2': (((0, 1, 1, 0, 0, 1, 0, 1), 1.0),),
    'x3': (((0, 0, 0, 1, 0, 1, 1, 1), 0.5), ((0, 0, 0, 0, 0, 0, 0, 1), 0.5)),
}


class ProbabilisticNodeElement(Element):
    """Node of a probabilistic Boolean network, on or off.

    It holds the index of the function that set it last.

    Args:
        label:
            The label of the node.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, 0 or 1.

        function:
            The index of the candidate function of the last update, 0 at the start.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import ProbabilisticNodeElement
        >>> ProbabilisticNodeElement('x1', x=1).x
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        function: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.function = function


class ProbabilisticFunctionInteraction(MapInteraction):
    """Synchronous map of a node by one of several truth tables.

    The table is drawn with given probabilities at every step.

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
        >>> from skcomplex.systems.biology.systems_biology import ProbabilisticFunctionInteraction
        >>> function = ProbabilisticFunctionInteraction(
        ...     'f', target='c', inputs=['a', 'b'], tables=((0, 0, 0, 1), (0, 1, 1, 1)), probabilities=(0.7, 0.3)
        ... )
        >>> function.probabilities
        (0.7, 0.3)
    """

    draws = True

    law_classes = (UpdateLaw,)

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
        """Return the product that is 1 when the inputs spell an index.

        The inputs are read as bits from the most significant.
        """
        bits = [(index >> (len(inputs) - 1 - position)) & 1 for position in range(len(inputs))]
        return prod((value if bit else 1 - value for value, bit in zip(inputs, bits, strict=True)), start=SR(1))

    def _get_table_value(self, inputs: list[Expression], table: Sequence[int]) -> Expression:
        """Return the value of one truth table of the inputs."""
        return sum((self._get_pattern(inputs, index) for index, entry in enumerate(table) if entry), start=SR(0))

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for invalid tables and probabilities.

        They are invalid when they differ in number, or a table has the wrong size.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The tables and the probabilities differ in number, or a table does not have one entry per pattern.
        """
        super()._prepare(context)
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
        """Return the index of the table drawn in a step.

        One draw serves the value and its record.
        """
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
        """Return the map of the target by the drawn table.

        It also returns the record of the drawn index.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of the value and the update of the index of the drawn table.
        """
        return [*super()._get_laws(namespace), UpdateLaw(namespace.target.function, self._get_draw())]


class ProbabilisticBooleanNetworkSystem(BaseSystem):
    """A probabilistic Boolean network.

    Its nodes draw one of their candidate functions at every step.

    Every node has candidate truth tables of the inputs, each with a probability, and at every step every node draws one
    of its tables independently and takes its value, all nodes together. The default network follows the three-gene
    example of Shmulevich et al., in which `x1` and `x3` have two candidate functions and `x2` one. From one initial
    state the network reaches different attractors in different runs. The system has the values `<node>.x` and the
    indices of the drawn functions `<node>.function`.

    Args:
        network:
            A mapping from node name to its candidate tables with their probabilities, or `None` for the three-gene
            example.

        inputs:
            The nodes every table reads, in the order of its bits, from the most significant.

        active:
            The nodes that are on at the start.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import ProbabilisticBooleanNetworkSystem
        >>> ProbabilisticBooleanNetworkSystem().prepare().system_.interactions[0].probabilities
        (0.6, 0.4)
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'gene regulation', 'discrete map', 'stochastic')
    references = ('Shmulevich et al. 2002', 'Raman 2021')
    until = 100
    dt = 1

    def __init__(
        self,
        network: Mapping[str, Candidates] | None = None,
        *,
        inputs: Sequence[str] = SHMULEVICH_INPUTS,
        active: Sequence[str] = ('x1',),
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.network = network
        self.inputs = inputs
        self.active = active

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a probabilistic Boolean network."""
        chosen = SHMULEVICH_NETWORK if self.network is None else self.network
        nodes = [ProbabilisticNodeElement(name, x=int(name in self.active)) for name in chosen]
        functions = [
            ProbabilisticFunctionInteraction(
                f'functions_{name}',
                target=name,
                inputs=list(self.inputs),
                tables=tuple(tuple(table) for table, _ in candidates),
                probabilities=tuple(probability for _, probability in candidates),
            )
            for name, candidates in chosen.items()
        ]
        return CompositeSystem(self.label, elements=nodes, interactions=functions)
