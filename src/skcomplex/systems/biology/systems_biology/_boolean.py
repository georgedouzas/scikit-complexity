"""Systems biology systems: Boolean gene regulatory networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import where
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec
from ..._catalog import register_system
from ._boolean_function import BooleanFunctionInteraction
from ._boolean_node import BooleanNodeElement

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


class ThresholdFunctionInteraction(MapInteraction):
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
        >>> from skcomplex.systems.biology.systems_biology import ThresholdFunctionInteraction
        >>> ThresholdFunctionInteraction('sic1', target='Sic1', inputs=['Swi5', 'Cln1_2'], weights=[1.0, -1.0]).weights
        [1.0, -1.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        weights: Annotated[Sequence[float], ParameterSpec(per='inputs')] = (),
        decays: bool = False,
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._get_next)
        self.inputs = inputs
        self.weights = weights
        self.decays = decays

    def _get_next(self, s: SimpleNamespace) -> Expression:
        """Return 1 for a positive weighted input, 0 for a negative one, and the held or decayed value for none."""
        total = sum((weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True)), start=SR(0))
        unchanged = 0 if self.decays else s.target.x
        return where(total > 0, 1, where(total < 0, 0, unchanged))


@register_system(
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
            The system, with the values `node_<i>.x`.

    Raises:
        ValidationError:
            `k` is not between 1 and the number of nodes.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import random_boolean_network
        >>> len(random_boolean_network(10, 3).interactions[0].table)
        8
    """
    if not 1 <= k <= n_nodes:
        error_msg = f'k = {k} inputs per node must be between 1 and the number of nodes, n_nodes = {n_nodes}.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    nodes = [BooleanNodeElement(f'node_{index}', x=int(rng.integers(2))) for index in range(n_nodes)]
    functions = [
        BooleanFunctionInteraction(
            f'function_{index}',
            target=f'node_{index}',
            inputs=[f'node_{source}' for source in rng.choice(n_nodes, size=k, replace=False)],
            table=tuple(int(entry) for entry in rng.random(2**k) < bias),
        )
        for index in range(n_nodes)
    ]
    return System(elements=nodes, interactions=functions)


@register_system(
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
            The system, with the values `<protein>.x`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import yeast_cell_cycle
        >>> [node.label for node in yeast_cell_cycle().elements if node.x]
        ['Cln3', 'Cdh1', 'Sic1']
    """
    nodes = [BooleanNodeElement(name, x=int(name in active)) for name in YEAST_NODES]
    functions = [
        ThresholdFunctionInteraction(
            f'regulation_{name}',
            target=name,
            inputs=[source for source, _ in YEAST_INPUTS[name]],
            weights=[float(weight) for _, weight in YEAST_INPUTS[name]],
            decays=name in YEAST_DECAYING,
        )
        for name in YEAST_NODES
    ]
    return System(elements=nodes, interactions=functions)
