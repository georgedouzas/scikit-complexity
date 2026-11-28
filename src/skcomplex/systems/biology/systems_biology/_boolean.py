"""Systems biology systems: Boolean gene regulatory networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, CallSpec, UpdateLaw
from ....core.components.laws.utils import where
from ....core.engine.backends import BaseBackend, NumpyBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import IntArray, Role
from ..common.networks import BooleanNodeElement

BOOLEAN = FiniteEnumeratedSet([0, 1])
UPDATES = ('synchronous', 'asynchronous')
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


class ThresholdFunctionInteraction(Interaction):
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

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        inputs: tuple[Role, ...],
        weights: Sequence[float] = (),
        decays: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights
        self.decays = decays

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of a node from its weighted input.

        The value is 1 for a positive weighted input, 0 for a negative one, and the held or decayed value for none.

        Args:
            context:
                The namespace of the node and its inputs.
        """
        super()._prepare(context)
        weights = context.weights if self.inputs else ()
        total = sum((weight * item.x for weight, item in zip(weights, context.inputs, strict=True)), start=SR(0))
        unchanged = 0 if self.decays else context.target.x
        self.laws_: list[BaseLaw] = [
            UpdateLaw(context.target.x, where(total > 0, 1, where(total < 0, 0, unchanged))),
        ]


class RandomBooleanNodeElement(Element):
    """Nodes of a random Boolean network, each with its own inputs and truth table.

    Args:
        label:
            The label of the group of nodes.

        x:
            The value of every node, 0 or 1.

        inputs:
            The indices of the `k` nodes every node reads, one row per node, or 0 for a network that draws them anew
            at every step.

        table:
            The next value of every node for each of the `2^k` patterns of its inputs, read as the bits of a binary
            number from the most significant, one row per node, or 0 for a network that draws them anew at every step.

        size:
            The number of nodes, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import RandomBooleanNodeElement
        >>> nodes = RandomBooleanNodeElement('nodes', x=[0, 1], inputs=[[1], [0]], table=[[1, 0], [1, 0]], size=2)
        >>> nodes.table
        [[1, 0], [1, 0]]
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, BOOLEAN] = 0,
        inputs: Annotated[object, ZZ] = 0,
        table: Annotated[object, BOOLEAN] = 0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.inputs = inputs
        self.table = table


class TruthTablesInteraction(Interaction):
    """Update of every node of a group by its own truth table of its inputs.

    Node `i` takes the entry of its table at the pattern of its `k` inputs, read as the bits of a binary number from the
    most significant. A synchronous update sets all nodes at once from the values at the start of the step. An
    asynchronous update draws one node uniformly and sets it from the current values, `n` times per step for `n` nodes,
    so one unit of time holds `n` single updates. An annealed network draws new inputs and tables at every step, `k`
    distinct inputs per node and entries that are 1 with probability `bias`, instead of reading those of the nodes.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of nodes.

        k:
            The number of inputs per node.

        update:
            `'synchronous'` to set all nodes at once or `'asynchronous'` to set one random node at a time.

        annealed:
            Whether to draw new inputs and tables at every step.

        bias:
            The probability that an entry of a drawn table is 1.

    Raises:
        ValidationError:
            The update is neither synchronous nor asynchronous.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import TruthTablesInteraction
        >>> TruthTablesInteraction('functions', target='nodes', k=3, update='asynchronous').update
        'asynchronous'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'nodes',
        k: int = 2,
        update: str = 'synchronous',
        annealed: bool = False,
        bias: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.update = update
        self.annealed = annealed
        self.bias = bias

    def _check_arguments(self) -> None:
        """Raise an error for an update the interaction cannot take.

        Raises:
            ValidationError:
                The update is neither synchronous nor asynchronous.
        """
        super()._check_arguments()
        if self.update not in UPDATES:
            error_msg = f'Unknown update {self.update!r} of {self.label!r}. Use one of {", ".join(UPDATES)}.'
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next values of the nodes by their tables.

        The tables are those of the nodes, or drawn tables for an annealed network.

        Args:
            context:
                The namespace of the group of nodes.
        """
        super()._prepare(context)
        nodes = context.target
        random = self.annealed or self.update == 'asynchronous'
        arguments = (nodes.x,) if self.annealed else (nodes.x, *nodes.inputs, *nodes.table)
        call = CallSpec(self._get_next, arguments, random=random, vectorized=True)
        self.laws_: list[BaseLaw] = [UpdateLaw(nodes.x, call)]

    def _get_next(
        self,
        t: float,
        x: npt.ArrayLike,
        *columns: npt.ArrayLike,
        rng: np.random.Generator | None = None,
    ) -> IntArray:
        """Return the next values of the nodes.

        The values may come with leading axes of configurations, which a synchronous update keeps.
        """
        values = np.asarray(x).astype(np.int64)
        size = values.shape[-1]
        if self.annealed and rng is not None:
            inputs, table = self._draw(size, rng)
        else:
            inputs = np.stack([np.asarray(column) for column in columns[: self.k]], axis=-1).astype(np.int64)
            table = np.stack([np.asarray(column) for column in columns[self.k :]], axis=-1).astype(np.int64)
        weights = 2 ** np.arange(self.k - 1, -1, -1)
        if self.update == 'synchronous' or rng is None:
            return table[np.arange(size), values[..., inputs] @ weights]
        values = values.copy()
        for node in rng.integers(size, size=(size,)):
            values[node] = table[node, values[inputs[node]] @ weights]
        return values

    def _draw(self, size: int, rng: np.random.Generator) -> tuple[IntArray, IntArray]:
        """Return `k` distinct inputs and a table per node, drawn anew.

        Rows with a repeated input are drawn again, or the inputs are a random order of all nodes when `k` is above
        half of them.
        """
        table = (rng.random((size, 2**self.k)) < self.bias).astype(np.int64)
        if 2 * self.k > size:
            return rng.random((size, size)).argsort(axis=1)[:, : self.k], table
        inputs = np.empty((size, self.k), dtype=np.int64)
        repeated = np.full(size, True)
        while repeated.any():
            inputs[repeated] = rng.integers(size, size=(int(repeated.sum()), self.k))
            repeated = (np.diff(np.sort(inputs, axis=1), axis=1) == 0).any(axis=1)
        return inputs, table


class RandomBooleanNetworkSystem(BaseSystem):
    """A Kauffman random Boolean network.

    Its nodes are updated by random functions of `k` random inputs.

    Each node reads `k` distinct nodes, possibly itself, through a truth table whose entries are 1 with probability
    `bias`. The quenched network keeps its inputs and tables, so its finite state space makes every synchronous run end
    on a cycle. With `k = 2` and `bias = 0.5` the network is critical, between frozen and chaotic. An asynchronous
    update sets one node drawn uniformly at a time, `n_nodes` single updates per unit of time. An annealed network draws
    new inputs and tables at every step, the mean field of Derrida and Pomeau. The system has the group `nodes` with the
    state `nodes.x` of shape `(n_nodes,)`, and the parameters `nodes.inputs` and `nodes.table` of a quenched network.

    Args:
        n_nodes:
            The number of nodes.

        k:
            The number of inputs per node, at most the number of nodes.

        bias:
            The probability that an entry of a truth table is 1.

        update:
            `'synchronous'` to set all nodes at once or `'asynchronous'` to set one random node at a time.

        annealed:
            Whether to draw new inputs and tables at every step.

        random_state:
            The seed or generator of the wiring, the tables, and the initial values.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            `k` is not between 1 and the number of nodes, or the update is neither synchronous nor asynchronous.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import RandomBooleanNetworkSystem
        >>> nodes = RandomBooleanNetworkSystem(10, 3).prepare().system_.elements[0]
        >>> nodes.size, len(nodes.table[0])
        (10, 8)
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'gene regulation', 'discrete map', 'criticality')
    references = ('Kauffman 1969', 'Derrida and Pomeau 1986', 'Gros 2015')
    until = 50
    dt = 1

    def __init__(
        self,
        n_nodes: int = 20,
        k: int = 2,
        *,
        bias: float = 0.5,
        update: str = 'synchronous',
        annealed: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.k = k
        self.bias = bias
        self.update = update
        self.annealed = annealed
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a Kauffman random Boolean network.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                `k` is not between 1 and the number of nodes.
        """
        super()._prepare(context)
        if not 1 <= self.k <= self.n_nodes:
            error_msg = (
                f'k = {self.k} inputs per node must be between 1 and the number of nodes, n_nodes = {self.n_nodes}.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        x = rng.integers(2, size=self.n_nodes)
        if self.annealed:
            nodes = RandomBooleanNodeElement('nodes', x=x, size=self.n_nodes)
        else:
            inputs = np.stack([rng.choice(self.n_nodes, size=self.k, replace=False) for _ in range(self.n_nodes)])
            table = (rng.random((self.n_nodes, 2**self.k)) < self.bias).astype(np.int64)
            nodes = RandomBooleanNodeElement('nodes', x=x, inputs=inputs, table=table, size=self.n_nodes)
        functions = TruthTablesInteraction(
            'functions',
            target='nodes',
            k=self.k,
            update=self.update,
            annealed=self.annealed,
            bias=self.bias,
        )
        self.system_ = System(self.label, elements=[nodes], interactions=[functions], backend=NumpyBackend())


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.active = active

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Boolean network of the yeast cell cycle of Li et al.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        nodes = [BooleanNodeElement(name, x=int(name in self.active)) for name in YEAST_NODES]
        functions = [
            ThresholdFunctionInteraction(
                f'regulation_{name}',
                target=name,
                inputs=tuple(source for source, _ in YEAST_INPUTS[name]),
                weights=[float(weight) for _, weight in YEAST_INPUTS[name]],
                decays=name in YEAST_DECAYING,
            )
            for name in YEAST_NODES
        ]
        self.system_ = System(self.label, elements=nodes, interactions=functions)
