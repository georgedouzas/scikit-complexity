"""Systems biology models: probabilistic Boolean networks, which draw the function of every node at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from sage.all__sagemath_symbolics import SR

from ....core.base import Domain, Expression, ValidationError, role, state
from ....core.model import Element, Interaction, Law, Symbols, System, Update, categorical, where
from ....core.runtime import register_model
from ._boolean import _pattern

type Candidates = Sequence[tuple[Sequence[int], float]]

SHMULEVICH_INPUTS = ('x1', 'x2', 'x3')
SHMULEVICH_NETWORK: dict[str, tuple[tuple[tuple[int, ...], float], ...]] = {
    'x1': (((0, 1, 1, 1, 0, 1, 1, 1), 0.6), ((0, 0, 0, 1, 0, 1, 1, 1), 0.4)),
    'x2': (((0, 1, 1, 0, 0, 1, 0, 1), 1.0),),
    'x3': (((0, 0, 0, 1, 0, 1, 1, 1), 0.5), ((0, 0, 0, 0, 0, 0, 0, 1), 0.5)),
}


class ProbabilisticNode(Element):
    """Node of a probabilistic Boolean network, on or off, with the index of the function that set it last.

    Args:
        label:
            The label of the node.

        x:
            The initial value, 0 or 1.

        function:
            The index of the candidate function of the last update, 0 at the start.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ProbabilisticNode
        >>> ProbabilisticNode('x1', x=1).x
        1
    """

    x: float = state(default=0, domain=Domain.BOOLEAN)
    function: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)


class ProbabilisticFunction(Interaction):
    """Synchronous update of a node by one of several truth tables, drawn with given probabilities at every step.

    Each table of `2^k` entries reads the `k` inputs as the bits of a binary number, from the most significant, and
    gives the next value of the target. The target also records the index of the drawn table.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node, a `ProbabilisticNode`.

        inputs:
            The paths of the input nodes, in the order of the bits of the tables.

        tables:
            The candidate truth tables.

        probabilities:
            The probability of every table.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ProbabilisticFunction
        >>> function = ProbabilisticFunction(
        ...     'f', target='c', inputs=['a', 'b'], tables=((0, 0, 0, 1), (0, 1, 1, 1)), probabilities=(0.7, 0.3)
        ... )
        >>> function.probabilities
        (0.7, 0.3)
    """

    target: str = role()
    inputs: list[str] = role(variadic=True)
    tables: Sequence[Sequence[int]] = ((0,),)
    probabilities: Sequence[float] = (1.0,)

    def _table_value(self, inputs: list[Expression], table: Sequence[int]) -> Expression:
        """Return the value of one truth table of the inputs."""
        if len(table) != 2 ** len(inputs):
            error_msg = (
                f'ProbabilisticFunction {self.label} reads {len(inputs)} inputs, which take {2 ** len(inputs)} '
                f'patterns, but one of its tables has {len(table)} entries.'
            )
            raise ValidationError(error_msg)
        return sum((_pattern(inputs, index) for index, entry in enumerate(table) if entry), start=SR(0))

    def laws(self, s: Symbols) -> list[Law]:
        """Return the update of the target by the drawn table and the record of the drawn index.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The tables and the probabilities differ in number, or a table does not have one entry per pattern.
        """
        if len(self.tables) != len(self.probabilities):
            error_msg = (
                f'ProbabilisticFunction {self.label} has {len(self.tables)} tables and {len(self.probabilities)} '
                'probabilities. Give one probability per table.'
            )
            raise ValidationError(error_msg)
        inputs = [item.x for item in s.inputs]
        chosen = categorical('function', list(self.probabilities))
        values = [self._table_value(inputs, table) for table in self.tables]
        value = values[-1]
        for index in range(len(values) - 2, -1, -1):
            value = where(chosen == index, values[index], value)
        return [Update(s.target.x, value), Update(s.target.function, chosen)]


@register_model(
    field='biology/systems_biology',
    tags=('boolean network', 'gene regulation', 'discrete map', 'stochastic'),
    references=('Shmulevich et al. 2002', 'Raman 2021'),
    defaults={'until': 100, 'dt': 1},
)
def probabilistic_boolean_network(
    network: Mapping[str, Candidates] | None = None,
    *,
    inputs: Sequence[str] = SHMULEVICH_INPUTS,
    active: Sequence[str] = ('x1',),
) -> System:
    """Return a probabilistic Boolean network, whose nodes draw one of their candidate functions at every step.

    Every node has candidate truth tables of the inputs, each with a probability, and at every step every node draws
    one of its tables independently and takes its value, all nodes together. The default network follows the three-gene
    example of Shmulevich et al., in which `x1` and `x3` have two candidate functions and `x2` one. From one initial
    state the network reaches different attractors in different runs.

    Args:
        network:
            A mapping from node name to its candidate tables with their probabilities, or `None` for the three-gene
            example.

        inputs:
            The nodes every table reads, in the order of its bits, from the most significant.

        active:
            The nodes that are on at the start.

    Returns:
        system:
            The model, with the values `<node>.x` and the indices of the drawn functions `<node>.function`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import probabilistic_boolean_network
        >>> probabilistic_boolean_network().interactions[0].probabilities
        (0.6, 0.4)
    """
    chosen = SHMULEVICH_NETWORK if network is None else network
    nodes = [ProbabilisticNode(name, x=int(name in active)) for name in chosen]
    functions = [
        ProbabilisticFunction(
            f'functions_{name}',
            target=name,
            inputs=list(inputs),
            tables=tuple(tuple(table) for table, _ in candidates),
            probabilities=tuple(probability for _, probability in candidates),
        )
        for name, candidates in chosen.items()
    ]
    return System(components=nodes, interactions=functions)
