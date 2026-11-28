"""Boolean networks of gene circuits, the mammalian cell cycle, and noisy majorities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from math import prod
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import total, uniform, where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.networks import BooleanNodeElement
from .types import Rule

RAMAN = 'Raman 2021'
THREE_GENE_RULES: dict[str, tuple[tuple[str, ...], Rule]] = {
    'g1': (('g2', 'g3'), lambda g2, g3: g2 and not g3),
    'g2': (('g1',), lambda g1: not g1),
    'g3': (('g2', 'g3'), lambda g2, g3: not g3 or g2),
}
CELL_CYCLE_RULES: dict[str, tuple[tuple[str, ...], Rule]] = {
    'CycD': (('CycD',), lambda cycd: cycd),
    'Rb': (
        ('CycA', 'CycB', 'CycD', 'CycE', 'p27'),
        lambda cyca, cycb, cycd, cyce, p27: (not cyca and not cycb and not cycd and not cyce)
        or (p27 and not cycb and not cycd),
    ),
    'E2F': (
        ('Rb', 'CycA', 'CycB', 'p27'),
        lambda rb, cyca, cycb, p27: (not rb and not cyca and not cycb) or (p27 and not rb and not cycb),
    ),
    'CycE': (('E2F', 'Rb'), lambda e2f, rb: e2f and not rb),
    'CycA': (
        ('E2F', 'Rb', 'Cdc20', 'Cdh1', 'Ubc', 'CycA'),
        lambda e2f, rb, cdc20, cdh1, ubc, cyca: (e2f or cyca) and not rb and not cdc20 and not (cdh1 and ubc),
    ),
    'p27': (
        ('CycD', 'CycE', 'CycA', 'CycB', 'p27'),
        lambda cycd, cyce, cyca, cycb, p27: (not cycd and not cyce and not cyca and not cycb)
        or (p27 and not (cyce and cyca) and not cycb and not cycd),
    ),
    'Cdc20': (('CycB',), lambda cycb: cycb),
    'Cdh1': (
        ('CycA', 'CycB', 'Cdc20', 'p27'),
        lambda cyca, cycb, cdc20, p27: (not cyca and not cycb) or cdc20 or (p27 and not cycb),
    ),
    'Ubc': (
        ('Cdh1', 'Ubc', 'Cdc20', 'CycA', 'CycB'),
        lambda cdh1, ubc, cdc20, cyca, cycb: not cdh1 or (cdh1 and ubc and (cdc20 or cyca or cycb)),
    ),
    'CycB': (('Cdc20', 'Cdh1'), lambda cdc20, cdh1: not cdc20 and not cdh1),
}
CELL_CYCLE_START = ('CycD', 'Rb', 'p27', 'Cdh1')
REPRESSILATOR_GENES = ('LacI', 'TetR', 'cI')


class BooleanFunctionInteraction(Interaction):
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
        >>> from skcomplex.systems.biology.systems_biology import BooleanFunctionInteraction
        >>> BooleanFunctionInteraction('and', target='c', inputs=['a', 'b'], table=(0, 0, 0, 1)).table
        (0, 0, 0, 1)
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        inputs: tuple[Role, ...],
        table: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.table = table

    def _get_pattern(self, inputs: Sequence[Expression], index: int) -> Expression:
        """Return the product that is 1 when the inputs spell an index.

        The inputs are read as bits from the most significant.
        """
        bits = [(index >> (len(inputs) - 1 - position)) & 1 for position in range(len(inputs))]
        return prod((value if bit else 1 - value for value, bit in zip(inputs, bits, strict=True)), start=SR(1))

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The table does not have one entry per pattern of the inputs.
        """
        super()._check_arguments()
        if len(self.table) != 2 ** len(self.inputs):
            error_msg = (
                f'BooleanFunction {self.label} reads {len(self.inputs)} inputs, which take {2 ** len(self.inputs)} '
                f'patterns, but its table has {len(self.table)} entries.'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of the target by the truth table.

        Args:
            context:
                The namespace of the target and the inputs.
        """
        super()._prepare(context)
        inputs = [item.x for item in context.inputs]
        terms = [self._get_pattern(inputs, index) for index, entry in enumerate(self.table) if entry]
        self.laws_: list[BaseLaw] = [UpdateLaw(context.target.x, sum(terms, start=SR(0)))]


class MajorityNodeElement(Element):
    """Node of a noisy Boolean network, a state of plus or minus one.

    The node follows the majority of its inputs.

    Args:
        label:
            The label of the node or of the group of nodes.

        sigma:
            The state, -1 or 1.

        eta:
            The probability that the node takes the state opposite to the majority.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import MajorityNodeElement
        >>> MajorityNodeElement('node', eta=0.1).eta
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        sigma: Annotated[float | Array, FiniteEnumeratedSet([-1, 1])] = 1.0,
        eta: Annotated[float | Array, RealSet.closed(0, 1)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.sigma = sigma
        self.eta = eta


class NoisyMajorityInteraction(Interaction):
    """Update of every node of a Boolean network to the sign of the sum of its inputs.

    The sign is flipped with a probability.

    Node `i` takes `sign(sum_j sigma_ij)` over its inputs with the probability `1 - eta` and the opposite sign with
    the probability `eta`, all nodes at once, the network of Huepe and Aldana-González (2002).

    Args:
        label:
            The label of the interaction.

        nodes:
            The path of the group of nodes.

        inputs:
            The input matrix, dense or sparse, one row per node with 1 for every input, an odd number per row.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import NoisyMajorityInteraction
        >>> NoisyMajorityInteraction('majority', nodes='nodes', inputs=((0.0, 1.0), (1.0, 0.0))).nodes
        'nodes'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        nodes: Role,
        inputs: Sequence[Sequence[float]] | scipy.sparse.sparray = ((0.0,),),
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.inputs = inputs

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the noisy majority update.

        Args:
            context:
                The namespace of the group of nodes.
        """
        super()._prepare(context)
        node = context.nodes
        weights = self.inputs if scipy.sparse.issparse(self.inputs) else np.asarray(self.inputs, dtype=float)
        majority = where(total(node.sigma, weights=weights) > 0, 1, -1)
        self.laws_: list[BaseLaw] = [UpdateLaw(node.sigma, where(uniform('flip') < node.eta, -majority, majority))]


class ThreeGeneBooleanNetworkSystem(BaseSystem):
    """The three-gene Boolean network of Raman, updated synchronously.

    The genes follow `g1* = g2 AND NOT g3`, `g2* = NOT g1`, and `g3* = NOT g3 OR g2`. Its eight states fall into two
    basins: the point attractor `011` and the cycle `001 -> 010 -> 111` of period three, which the state `100` enters.
    The system has the values `g1.x`, `g2.x`, and `g3.x`.

    Args:
        x:
            The initial values of `g1`, `g2`, and `g3`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import ThreeGeneBooleanNetworkSystem
        >>> ThreeGeneBooleanNetworkSystem().prepare().system_.interactions[2].table
        (1, 0, 1, 1)
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'gene regulation', 'discrete map', 'attractors')
    references = (RAMAN,)
    until = 10
    dt = 1

    @staticmethod
    def _get_truth_table(rule: Rule, arity: int) -> tuple[int, ...]:
        """Return the next value of a rule for every pattern of its inputs.

        The patterns are read as bits from the most significant.
        """
        return tuple(int(bool(rule(*pattern))) for pattern in itertools.product((0, 1), repeat=arity))

    @staticmethod
    def _get_rule_network(
        label: str,
        rules: dict[str, tuple[tuple[str, ...], Rule]],
        active: Sequence[str],
        knockouts: Sequence[str] = (),
    ) -> System:
        """Return the labelled Boolean network of rules.

        The active nodes are on and the knocked-out nodes kept off.
        """
        unknown = sorted((set(active) | set(knockouts)) - set(rules))
        if unknown:
            error_msg = f'Unknown nodes {unknown}. Use names among {list(rules)}.'
            raise ValidationError(error_msg)
        nodes = [BooleanNodeElement(name, x=int(name in active and name not in knockouts)) for name in rules]
        functions = [
            BooleanFunctionInteraction(
                f'function_{name}',
                target=name,
                inputs=tuple(inputs),
                table=(
                    (0,) * 2 ** len(inputs)
                    if name in knockouts
                    else ThreeGeneBooleanNetworkSystem._get_truth_table(rule, len(inputs))
                ),
            )
            for name, (inputs, rule) in rules.items()
        ]
        return System(label, elements=nodes, interactions=functions)

    def __init__(
        self,
        x: Sequence[int] = (1, 0, 0),
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the three-gene Boolean network of Raman.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        active = [name for name, value in zip(THREE_GENE_RULES, self.x, strict=True) if value]
        self.system_ = ThreeGeneBooleanNetworkSystem._get_rule_network(self.label, THREE_GENE_RULES, active)


class MammalianCellCycleSystem(BaseSystem):
    """The Boolean system of the mammalian cell cycle of Faure et al.

    It holds ten proteins updated synchronously.

    The rules are those printed by Raman for CycD, Rb, E2F, CycE, CycA, p27, Cdc20, Cdh1, UbcH10 (`Ubc`), and CycB. The
    growth signal CycD keeps its value. Without it the network rests in the quiescent state with Rb, p27, and Cdh1 on.
    With it the network leaves that state and runs through a cycle of period 7 in which the cyclins E, A, and B rise and
    fall in turn. A knocked-out protein stays off. The system has the values `<protein>.x`.

    Args:
        active:
            The proteins that are on at the start.

        knockouts:
            The proteins that are knocked out.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            A name is not one of the ten proteins.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import MammalianCellCycleSystem
        >>> [node.label for node in MammalianCellCycleSystem().prepare().system_.elements if node.x]
        ['CycD', 'Rb', 'p27', 'Cdh1']
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'cell cycle', 'discrete map', 'attractors', 'knock-out')
    references = (RAMAN, 'Faure et al. 2006')
    until = 21
    dt = 1

    def __init__(
        self,
        active: Sequence[str] = CELL_CYCLE_START,
        *,
        knockouts: Sequence[str] = (),
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.active = active
        self.knockouts = knockouts

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Boolean system of the mammalian cell cycle.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = ThreeGeneBooleanNetworkSystem._get_rule_network(
            self.label,
            CELL_CYCLE_RULES,
            self.active,
            self.knockouts,
        )


class BooleanRepressilatorSystem(BaseSystem):
    """The Boolean repressilator of LacI, TetR, and cI.

    LacI represses TetR, TetR represses cI, and cI represses LacI.

    The transfer functions are `TetR* = NOT LacI`, `cI* = NOT TetR`, and `LacI* = NOT cI`. From a state with one or two
    genes on, the genes switch on and off in turn with period 6, and the states `000` and `111` alternate with period
    2. A repression delayed by `delay` steps passes through that many relay nodes, `<repressor>_1` and so on, which
    copy the repressor one step later and start at its initial value. The oscillation remains, with period
    `6 (delay + 1)`. The system has the values `LacI.x`, `TetR.x`, `cI.x`, and the relays `<repressor>_<k>.x`.

    Args:
        x:
            The initial values of LacI, TetR, and cI.

        delay:
            The number of steps by which every repression is delayed.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The delay is negative.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import BooleanRepressilatorSystem
        >>> [node.label for node in BooleanRepressilatorSystem(delay=1).prepare().system_.elements]
        ['LacI', 'TetR', 'cI', 'LacI_1', 'TetR_1', 'cI_1']
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'repressilator', 'oscillator', 'discrete map', 'delay')
    references = (RAMAN, 'Elowitz and Leibler 2000')
    until = 24
    dt = 1

    def __init__(
        self,
        x: Sequence[int] = (1, 0, 0),
        *,
        delay: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.delay = delay

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Boolean repressilator.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        if self.delay < 0:
            error_msg = f'The delay must be a nonnegative number of steps, not {self.delay}.'
            raise ValidationError(error_msg)
        genes = [
            BooleanNodeElement(name, x=int(value)) for name, value in zip(REPRESSILATOR_GENES, self.x, strict=True)
        ]
        relays = [
            BooleanNodeElement(f'{gene.label}_{step}', x=gene.x) for step in range(1, self.delay + 1) for gene in genes
        ]
        functions: list[BooleanFunctionInteraction] = []
        for repressor, target in zip(
            REPRESSILATOR_GENES,
            (*REPRESSILATOR_GENES[1:], REPRESSILATOR_GENES[0]),
            strict=True,
        ):
            chain = [repressor, *(f'{repressor}_{step}' for step in range(1, self.delay + 1))]
            functions.extend(
                BooleanFunctionInteraction(f'relay_{relay}', target=relay, inputs=(source,), table=(0, 1))
                for source, relay in itertools.pairwise(chain)
            )
            functions.append(
                BooleanFunctionInteraction(f'repression_{target}', target=target, inputs=(chain[-1],), table=(1, 0)),
            )
        self.system_ = System(self.label, elements=[*genes, *relays], interactions=functions)


class NoisyBooleanNetworkSystem(BaseSystem):
    """The Boolean network of Huepe and Aldana-González.

    Its nodes follow the noisy majority of `k` inputs.

    Every node takes the sign of the sum of its `k` random inputs, flipped with the probability `eta`. The order
    parameter `Psi`, the time average of the absolute mean state, undergoes a continuous phase transition at the noise
    `eta_c = (1 - 1 / Phi'(0)) / 2` of mean-field theory, with `Phi'(0) = k C(k - 1, (k - 1) / 2) / 2^(k - 1)`, which is
    1/6 for `k = 3`. Below it the network orders, above it the mean state falls to fluctuations of order `1 / sqrt(n)`.
    The system has the states `nodes.sigma`.

    Args:
        n:
            The number of nodes.

        k:
            The number of inputs of every node, odd.

        eta:
            The noise, the probability of a flip.

        random_state:
            The seed of the inputs.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The number of inputs is even.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import NoisyBooleanNetworkSystem
        >>> NoisyBooleanNetworkSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'biology/systems_biology'
    topics = ('boolean network', 'noise', 'phase transition', 'mean field', 'stochastic map')
    references = ('Huepe and Aldana-González 2002', 'Gros 2015')
    until = 200
    dt = 1

    def __init__(
        self,
        n: int = 1000,
        *,
        k: int = 3,
        eta: float = 0.1,
        random_state: int | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.k = k
        self.eta = eta
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Boolean network of Huepe and Aldana-González.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        if self.k % 2 == 0:
            error_msg = f'The number of inputs must be odd so that the majority is defined, not {self.k}.'
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        inputs = np.zeros((self.n, self.n))
        for node in range(self.n):
            inputs[node, rng.choice(np.delete(np.arange(self.n), node), size=self.k, replace=False)] = 1.0
        nodes = MajorityNodeElement('nodes', sigma=np.ones(self.n), eta=self.eta, size=self.n)
        majority = NoisyMajorityInteraction('majority', nodes='nodes', inputs=scipy.sparse.csr_array(inputs))
        self.system_ = System(self.label, elements=[nodes], interactions=[majority])
