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
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec
from ..common.networks import BooleanFunctionInteraction, BooleanNodeElement

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
    """Synchronous update of a node by the sign of the weighted sum of its active inputs.

    The update follows Li et al.

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
        """Return the next value of a node from its weighted input.

        The value is 1 for a positive weighted input, 0 for a negative one, and the held or decayed value for none.
        """
        total = sum((weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True)), start=SR(0))
        unchanged = 0 if self.decays else s.target.x
        return where(total > 0, 1, where(total < 0, 0, unchanged))


class RandomBooleanNetworkSystem(BaseSystem):
    """A Kauffman random Boolean network.

    Its nodes are updated together by random functions of `k` random inputs.

    Each node reads `k` distinct nodes, possibly itself, through a truth table whose entries are 1 with probability
    `bias`. The finite state space makes every run end on a cycle. With `k = 2` and `bias = 0.5` the network is
    critical, between frozen and chaotic. The system has the values `node_<i>.x`.

    Args:
        n_nodes:
            The number of nodes.

        k:
            The number of inputs per node, at most the number of nodes.

        bias:
            The probability that an entry of a truth table is 1.

        random_state:
            The seed or generator of the wiring, the tables, and the initial values.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            `k` is not between 1 and the number of nodes.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import RandomBooleanNetworkSystem
        >>> len(RandomBooleanNetworkSystem(10, 3).prepare().system_.interactions[0].table)
        8
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'gene regulation', 'discrete map', 'criticality')
    references = ('Kauffman 1969', 'Gros 2015')
    until = 50
    dt = 1

    def __init__(
        self,
        n_nodes: int = 20,
        k: int = 2,
        *,
        bias: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.k = k
        self.bias = bias
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of a Kauffman random Boolean network."""
        if not 1 <= self.k <= self.n_nodes:
            error_msg = (
                f'k = {self.k} inputs per node must be between 1 and the number of nodes, n_nodes = {self.n_nodes}.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        nodes = [BooleanNodeElement(f'node_{index}', x=int(rng.integers(2))) for index in range(self.n_nodes)]
        functions = [
            BooleanFunctionInteraction(
                f'function_{index}',
                target=f'node_{index}',
                inputs=[f'node_{source}' for source in rng.choice(self.n_nodes, size=self.k, replace=False)],
                table=tuple(int(entry) for entry in rng.random(2**self.k) < self.bias),
            )
            for index in range(self.n_nodes)
        ]
        return CompositeSystem(self.label, elements=nodes, interactions=functions)


class YeastCellCycleSystem(BaseSystem):
    """The Boolean network of the budding yeast cell cycle of Li et al.

    It holds eleven proteins and their regulation.

    Every protein turns on or off with the sign of the summed activations and inhibitions by the active proteins, and
    the proteins Cln3, Cln1,2, Swi5, Cdc20, and Mcm1 decay without input. Started from the G1 state with Cln3 on, the
    cell size signal, the network runs through the phases of the cycle into the stationary G1 state. The system has the
    values `<protein>.x`.

    Args:
        active:
            The proteins that are on at the start, among `Cln3`, `MBF`, `SBF`, `Cln1_2`, `Cdh1`, `Swi5`, `Cdc20`,
            `Clb5_6`, `Sic1`, `Clb1_2`, and `Mcm1`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import YeastCellCycleSystem
        >>> [node.label for node in YeastCellCycleSystem().prepare().system_.elements if node.x]
        ['Cln3', 'Cdh1', 'Sic1']
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'gene regulation', 'discrete map', 'cell cycle')
    references = ('Li et al. 2004', 'Gros 2015')
    until = 15
    dt = 1

    def __init__(
        self,
        active: Sequence[str] = YEAST_START,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.active = active

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the Boolean network of the yeast cell cycle of Li et al.

        The yeast is the budding yeast.
        """
        nodes = [BooleanNodeElement(name, x=int(name in self.active)) for name in YEAST_NODES]
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
        return CompositeSystem(self.label, elements=nodes, interactions=functions)
