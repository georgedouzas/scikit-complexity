"""Systems biology models: Boolean gene regulatory networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from math import prod

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....core.base import Domain, Expression, ValidationError, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Symbols, Update, where
from ....core.runtime import register_model
from ....core.system import System

YEAST_NODES = ('Cln3', 'MBF', 'SBF', 'Cln1_2', 'Cdh1', 'Swi5', 'Cdc20', 'Clb5_6', 'Sic1', 'Clb1_2', 'Mcm1')
YEAST_INPUTS = {
    'Cln3': (),
    'MBF': (('Cln3', 1), ('Clb1_2', -1)),
    'SBF': (('Cln3', 1), ('Clb1_2', -1)),
    'Cln1_2': (('SBF', 1),),
    'Cdh1': (('Cln1_2', -1), ('Clb5_6', -1), ('Clb1_2', -1), ('Cdc20', 1)),
    'Swi5': (('Clb1_2', -1), ('Mcm1', 1), ('Cdc20', 1)),
    'Cdc20': (('Clb1_2', 1), ('Mcm1', 1)),
    'Clb5_6': (('MBF', 1), ('Cdc20', -1), ('Sic1', -1)),
    'Sic1': (('Cln1_2', -1), ('Clb5_6', -1), ('Clb1_2', -1), ('Cdc20', 1), ('Swi5', 1)),
    'Clb1_2': (('Clb5_6', 1), ('Mcm1', 1), ('Cdc20', -1), ('Cdh1', -1), ('Sic1', -1)),
    'Mcm1': (('Clb5_6', 1), ('Clb1_2', 1)),
}
YEAST_DECAYING = ('Cln3', 'Cln1_2', 'Swi5', 'Cdc20', 'Mcm1')
YEAST_START = ('Cln3', 'Cdh1', 'Sic1')


def _pattern(inputs: Sequence[Expression], index: int) -> Expression:
    """Return the product that is 1 when the inputs, read as bits from the most significant, spell an index."""
    bits = [(index >> (len(inputs) - 1 - position)) & 1 for position in range(len(inputs))]
    return prod((value if bit else 1 - value for value, bit in zip(inputs, bits, strict=True)), start=SR(1))


class BooleanNode(Element):
    """Node of a Boolean network, a gene or protein that is on or off.

    Args:
        label:
            The label of the node.

        x:
            The initial value, 0 or 1.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import BooleanNode
        >>> BooleanNode('Sic1', x=1).x
        1
    """

    x: float = state(default=0, domain=Domain.BOOLEAN)


class BooleanFunction(Interaction):
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

    Examples:
        >>> from skcomplex.models.biology.systems_biology import BooleanFunction
        >>> BooleanFunction('and', target='c', inputs=['a', 'b'], table=(0, 0, 0, 1)).table
        (0, 0, 0, 1)
    """

    target: str = role()
    inputs: list[str] = role(variadic=True)
    table: tuple[int, ...] = (0,)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the update of the target by the truth table.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The table does not have one entry per pattern of the inputs.
        """
        if len(self.table) != 2 ** len(s.inputs):
            error_msg = (
                f'BooleanFunction {self.label} reads {len(s.inputs)} inputs, which take {2 ** len(s.inputs)} '
                f'patterns, but its table has {len(self.table)} entries.'
            )
            raise ValidationError(error_msg)
        inputs = [item.x for item in s.inputs]
        terms = [_pattern(inputs, index) for index, entry in enumerate(self.table) if entry]
        return [Update(s.target.x, sum(terms, start=SR(0)))]


class ThresholdFunction(Interaction):
    """Synchronous update of a node by the sign of the weighted sum of its active inputs, after Li et al.

    The node turns on when the sum `w_0 x_0 + w_1 x_1 + ...` over its inputs is positive and off when it is negative.
    When the sum is zero, a node that decays turns off and any other node keeps its value.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node.

        inputs:
            The paths of the input nodes.

        weights:
            The weight of every input, positive for activation and negative for inhibition.

        decays:
            Whether the node turns off without net input.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ThresholdFunction
        >>> ThresholdFunction('sic1', target='Sic1', inputs=['Swi5', 'Cln1_2'], weights=[1.0, -1.0]).weights
        [1.0, -1.0]
    """

    target: str = role()
    inputs: list[str] = role(variadic=True)
    weights: list[float] = parameter(default=(), per='inputs')
    decays: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the threshold update of the target.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        total = sum((weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True)), start=SR(0))
        unchanged = 0 if self.decays else s.target.x
        return [Update(s.target.x, where(total > 0, 1, where(total < 0, 0, unchanged)))]


@register_model(
    field='biology/systems_biology',
    tags=('boolean network', 'gene regulation', 'discrete map', 'criticality'),
    references=('Kauffman 1969', 'Gros 2015'),
    defaults={'until': 50, 'dt': 1},
)
def random_boolean_network(
    n_nodes: int = 20,
    k: int = 2,
    *,
    bias: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a Kauffman random Boolean network, nodes updated together by random functions of `k` random inputs.

    Each node reads `k` distinct nodes, possibly itself, through a truth table whose entries are 1 with probability
    `bias`. The finite state space makes every run end on a cycle. With `k = 2` and `bias = 0.5` the network is
    critical, between frozen and chaotic.

    Args:
        n_nodes:
            The number of nodes.

        k:
            The number of inputs per node, at most the number of nodes.

        bias:
            The probability that an entry of a truth table is 1.

        random_state:
            The seed or generator of the wiring, the tables, and the initial values.

    Returns:
        system:
            The model, with the values `node_<i>.x`.

    Raises:
        ValidationError:
            `k` is not between 1 and the number of nodes.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import random_boolean_network
        >>> len(random_boolean_network(10, 3).interactions[0].table)
        8
    """
    if not 1 <= k <= n_nodes:
        error_msg = f'k = {k} inputs per node must be between 1 and the number of nodes, n_nodes = {n_nodes}.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    nodes = [BooleanNode(f'node_{index}', x=int(rng.integers(2))) for index in range(n_nodes)]
    functions = [
        BooleanFunction(
            f'function_{index}',
            target=f'node_{index}',
            inputs=[f'node_{source}' for source in rng.choice(n_nodes, size=k, replace=False)],
            table=tuple(int(entry) for entry in rng.random(2**k) < bias),
        )
        for index in range(n_nodes)
    ]
    return System(components=nodes, interactions=functions)


@register_model(
    field='biology/systems_biology',
    tags=('boolean network', 'gene regulation', 'discrete map', 'cell cycle'),
    references=('Li et al. 2004', 'Gros 2015'),
    defaults={'until': 15, 'dt': 1},
)
def yeast_cell_cycle(active: Sequence[str] = YEAST_START) -> System:
    """Return the Boolean network of the budding yeast cell cycle of Li et al., eleven proteins and their regulation.

    Every protein turns on or off with the sign of the summed activations and inhibitions by the active proteins,
    and the proteins Cln3, Cln1,2, Swi5, Cdc20, and Mcm1 decay without input. Started from the G1 state with Cln3 on,
    the cell size signal, the network runs through the phases of the cycle into the stationary G1 state.

    Args:
        active:
            The proteins that are on at the start, among `Cln3`, `MBF`, `SBF`, `Cln1_2`, `Cdh1`, `Swi5`, `Cdc20`,
            `Clb5_6`, `Sic1`, `Clb1_2`, and `Mcm1`.

    Returns:
        system:
            The model, with the values `<protein>.x`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import yeast_cell_cycle
        >>> [node.label for node in yeast_cell_cycle().elements if node.x]
        ['Cln3', 'Cdh1', 'Sic1']
    """
    nodes = [BooleanNode(name, x=int(name in active)) for name in YEAST_NODES]
    functions = [
        ThresholdFunction(
            f'regulation_{name}',
            target=name,
            inputs=[source for source, _ in YEAST_INPUTS[name]],
            weights=[float(weight) for _, weight in YEAST_INPUTS[name]],
            decays=name in YEAST_DECAYING,
        )
        for name in YEAST_NODES
    ]
    return System(components=nodes, interactions=functions)
