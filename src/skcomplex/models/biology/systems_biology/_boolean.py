"""Systems biology models: Boolean gene regulatory networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from math import prod

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....core.base import Domain, Expression, ValidationError, parameter, role, state
from ....core.model import Element, Interaction, Law, Symbols, System, Update, where
from ....core.runtime import register_model

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
INPUT_ROLES = ('input_0', 'input_1', 'input_2', 'input_3', 'input_4')


def _inputs_of(s: Symbols) -> list[Expression]:
    """Return the values of the five input roles of a Boolean update."""
    return [getattr(s, name).x for name in INPUT_ROLES]


def _pattern(inputs: Sequence[Expression], index: int) -> Expression:
    """Return the product that is 1 when the inputs, read as bits from the most significant, spell an index."""
    bits = [(index >> (len(inputs) - 1 - position)) & 1 for position in range(len(inputs))]
    return prod((value if bit else 1 - value for value, bit in zip(inputs, bits, strict=True)), start=SR(1))


def _padded_inputs(target: str, sources: Sequence[str]) -> list[str]:
    """Return the paths bound to the five input roles, the sources followed by the target for the unused roles."""
    if len(sources) > len(INPUT_ROLES):
        error_msg = f'A node reads at most {len(INPUT_ROLES)} inputs, and {target} reads {len(sources)}.'
        raise ValidationError(error_msg)
    return [*sources, *[target] * (len(INPUT_ROLES) - len(sources))]


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
    """Synchronous update of a node by a truth table of up to five input nodes.

    A table of `2^k` entries reads the first `k` inputs as the bits of a binary number, from the most significant,
    and gives the next value of the target. The update is the sum, over the patterns whose entry is 1, of the product
    that is 1 exactly on that pattern. Unused inputs are ignored, so they may be bound to any node.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node.

        input_0:
            The path of the first input.

        input_1:
            The path of the second input.

        input_2:
            The path of the third input.

        input_3:
            The path of the fourth input.

        input_4:
            The path of the fifth input.

        table:
            The next value of the target for each of the `2^k` input patterns.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import BooleanFunction
        >>> inputs = {'input_0': 'a', 'input_1': 'b', 'input_2': 'c', 'input_3': 'c', 'input_4': 'c'}
        >>> BooleanFunction('and', target='c', table=(0, 0, 0, 1), **inputs).table
        (0, 0, 0, 1)
    """

    target: str = role()
    input_0: str = role()
    input_1: str = role()
    input_2: str = role()
    input_3: str = role()
    input_4: str = role()
    table: tuple[int, ...] = (0,)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the update of the target by the truth table.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        inputs = _inputs_of(s)[: (len(self.table) - 1).bit_length()]
        terms = [_pattern(inputs, index) for index, entry in enumerate(self.table) if entry]
        return [Update(s.target.x, sum(terms, start=SR(0)))]


class ThresholdFunction(Interaction):
    """Synchronous update of a node by the sign of the weighted sum of its active inputs, after Li et al.

    The node turns on when the sum `w_0 x_0 + ... + w_4 x_4` is positive and off when it is negative. When the sum is
    zero, a node that decays turns off and any other node keeps its value. Inputs with zero weight are ignored.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the updated node.

        input_0:
            The path of the first input.

        input_1:
            The path of the second input.

        input_2:
            The path of the third input.

        input_3:
            The path of the fourth input.

        input_4:
            The path of the fifth input.

        w_0:
            The weight of the first input, positive for activation and negative for inhibition.

        w_1:
            The weight of the second input.

        w_2:
            The weight of the third input.

        w_3:
            The weight of the fourth input.

        w_4:
            The weight of the fifth input.

        decays:
            Whether the node turns off without net input.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ThresholdFunction
        >>> inputs = {'input_0': 'Swi5', 'input_1': 'Sic1', 'input_2': 'Sic1', 'input_3': 'Sic1', 'input_4': 'Sic1'}
        >>> ThresholdFunction('sic1', target='Sic1', w_0=1.0, **inputs).w_0
        1.0
    """

    target: str = role()
    input_0: str = role()
    input_1: str = role()
    input_2: str = role()
    input_3: str = role()
    input_4: str = role()
    w_0: float = parameter(default=0.0)
    w_1: float = parameter(default=0.0)
    w_2: float = parameter(default=0.0)
    w_3: float = parameter(default=0.0)
    w_4: float = parameter(default=0.0)
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
        weights = [s.w_0, s.w_1, s.w_2, s.w_3, s.w_4]
        total = sum((weight * value for weight, value in zip(weights, _inputs_of(s), strict=True)), start=SR(0))
        unchanged = 0 if self.decays else s.target.x
        return [Update(s.target.x, where(total > 0, 1, where(total < 0, 0, unchanged)))]


def _truth_table_update(label: str, target: str, sources: Sequence[str], table: tuple[int, ...]) -> BooleanFunction:
    """Return the truth table update of a target by its sources."""
    first, second, third, fourth, fifth = _padded_inputs(target, sources)
    return BooleanFunction(
        label,
        target=target,
        input_0=first,
        input_1=second,
        input_2=third,
        input_3=fourth,
        input_4=fifth,
        table=table,
    )


def _threshold_update(label: str, target: str, inputs: Sequence[tuple[str, int]], *, decays: bool) -> ThresholdFunction:
    """Return the threshold update of a target by weighted sources."""
    first, second, third, fourth, fifth = _padded_inputs(target, [source for source, _ in inputs])
    weights = [float(weight) for _, weight in inputs] + [0.0] * (len(INPUT_ROLES) - len(inputs))
    return ThresholdFunction(
        label,
        target=target,
        input_0=first,
        input_1=second,
        input_2=third,
        input_3=fourth,
        input_4=fifth,
        w_0=weights[0],
        w_1=weights[1],
        w_2=weights[2],
        w_3=weights[3],
        w_4=weights[4],
        decays=decays,
    )


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
            The number of inputs per node, at most five.

        bias:
            The probability that an entry of a truth table is 1.

        random_state:
            The seed or generator of the wiring, the tables, and the initial values.

    Returns:
        system:
            The model, with the values `node_<i>.x`.

    Raises:
        ValidationError:
            `k` exceeds five.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import random_boolean_network
        >>> len(random_boolean_network(10, 3).interactions[0].table)
        8
    """
    rng = np.random.default_rng(random_state)
    nodes = [BooleanNode(f'node_{index}', x=int(rng.integers(2))) for index in range(n_nodes)]
    functions = [
        _truth_table_update(
            f'function_{index}',
            f'node_{index}',
            [f'node_{source}' for source in rng.choice(n_nodes, size=k, replace=False)],
            tuple(int(entry) for entry in rng.random(2**k) < bias),
        )
        for index in range(n_nodes)
    ]
    return System(elements=nodes, interactions=functions)


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
        _threshold_update(f'regulation_{name}', name, YEAST_INPUTS[name], decays=name in YEAST_DECAYING)
        for name in YEAST_NODES
    ]
    return System(elements=nodes, interactions=functions)
