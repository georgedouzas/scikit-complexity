"""Systems biology systems: Boolean networks of gene circuits, the mammalian cell cycle, and noisy majorities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import scipy.sparse
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import total, uniform, where
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._boolean_function import BooleanFunctionInteraction
from ._boolean_node import BooleanNodeElement
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


class MajorityNodeElement(Element):
    """Node of a noisy Boolean network, a state of plus or minus one that follows the majority of its inputs.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sigma:
            The state, -1 or 1.

        eta:
            The probability that the node takes the state opposite to the majority.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import MajorityNodeElement
        >>> MajorityNodeElement('node', eta=0.1).eta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        sigma: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([-1, 1]))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sigma = sigma
        self.eta = eta


class NoisyMajorityInteraction(Interaction):
    """Update of every node of a Boolean network to the sign of the sum of its inputs, flipped with a probability.

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

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        inputs: Sequence[Sequence[float]] | scipy.sparse.sparray = ((0.0,),),
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.inputs = inputs

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the noisy majority update.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        node = s.nodes
        weights = self.inputs if scipy.sparse.issparse(self.inputs) else np.asarray(self.inputs, dtype=float)
        majority = where(total(node.sigma, weights=weights) > 0, 1, -1)
        return [UpdateLaw(node.sigma, where(uniform('flip') < node.eta, -majority, majority))]


def _truth_table(rule: Rule, arity: int) -> tuple[int, ...]:
    """Return the next value of a rule for every pattern of its inputs, read as bits from the most significant."""
    return tuple(int(bool(rule(*pattern))) for pattern in itertools.product((0, 1), repeat=arity))


def _rule_network(
    rules: dict[str, tuple[tuple[str, ...], Rule]],
    active: Sequence[str],
    knockouts: Sequence[str] = (),
) -> System:
    """Return the Boolean network of a set of rules, with the active nodes on and the knocked-out nodes kept off."""
    unknown = sorted((set(active) | set(knockouts)) - set(rules))
    if unknown:
        error_msg = f'Unknown nodes {unknown}. Use names among {list(rules)}.'
        raise ValidationError(error_msg)
    nodes = [BooleanNodeElement(name, x=int(name in active and name not in knockouts)) for name in rules]
    functions = [
        BooleanFunctionInteraction(
            f'function_{name}',
            target=name,
            inputs=list(inputs),
            table=(0,) * 2 ** len(inputs) if name in knockouts else _truth_table(rule, len(inputs)),
        )
        for name, (inputs, rule) in rules.items()
    ]
    return System(elements=nodes, interactions=functions)


@register_system(
    field='biology/systems_biology',
    tags=('boolean network', 'gene regulation', 'discrete map', 'attractors'),
    references=(RAMAN,),
    defaults={'until': 10, 'dt': 1},
)
def three_gene_boolean_network(x: Sequence[int] = (1, 0, 0)) -> System:
    """Return the three-gene Boolean network of Raman, updated synchronously.

    The genes follow `g1* = g2 AND NOT g3`, `g2* = NOT g1`, and `g3* = NOT g3 OR g2`. Its eight states fall into two
    basins: the point attractor `011` and the cycle `001 -> 010 -> 111` of period three, which the state `100` enters.

    Args:
        x:
            The initial values of `g1`, `g2`, and `g3`.

    Returns:
        system:
            The system, with the values `g1.x`, `g2.x`, and `g3.x`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import three_gene_boolean_network
        >>> three_gene_boolean_network().interactions[2].table
        (1, 0, 1, 1)
    """
    active = [name for name, value in zip(THREE_GENE_RULES, x, strict=True) if value]
    return _rule_network(THREE_GENE_RULES, active)


@register_system(
    field='biology/systems_biology',
    tags=('boolean network', 'cell cycle', 'discrete map', 'attractors', 'knock-out'),
    references=(RAMAN, 'Faure et al. 2006'),
    defaults={'until': 21, 'dt': 1},
)
def mammalian_cell_cycle(
    active: Sequence[str] = CELL_CYCLE_START,
    *,
    knockouts: Sequence[str] = (),
) -> System:
    """Return the Boolean system of the mammalian cell cycle of Faure et al., ten proteins updated synchronously.

    The rules are those printed by Raman for CycD, Rb, E2F, CycE, CycA, p27, Cdc20, Cdh1, UbcH10 (`Ubc`), and CycB.
    The growth signal CycD keeps its value. Without it the network rests in the quiescent state with Rb, p27, and Cdh1
    on. With it the network leaves that state and runs through a cycle of period 7 in which the cyclins E, A, and B
    rise and fall in turn. A knocked-out protein stays off.

    Args:
        active:
            The proteins that are on at the start.

        knockouts:
            The proteins that are knocked out.

    Returns:
        system:
            The system, with the values `<protein>.x`.

    Raises:
        ValidationError:
            A name is not one of the ten proteins.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import mammalian_cell_cycle
        >>> [node.label for node in mammalian_cell_cycle().elements if node.x]
        ['CycD', 'Rb', 'p27', 'Cdh1']
    """
    return _rule_network(CELL_CYCLE_RULES, active, knockouts)


@register_system(
    field='biology/systems_biology',
    tags=('boolean network', 'repressilator', 'oscillator', 'discrete map', 'delay'),
    references=(RAMAN, 'Elowitz and Leibler 2000'),
    defaults={'until': 24, 'dt': 1},
)
def boolean_repressilator(x: Sequence[int] = (1, 0, 0), *, delay: int = 0) -> System:
    """Return the Boolean repressilator, where LacI represses TetR, TetR represses cI, and cI represses LacI.

    The transfer functions are `TetR* = NOT LacI`, `cI* = NOT TetR`, and `LacI* = NOT cI`. From a state with one or two
    genes on, the genes switch on and off in turn with period 6, and the states `000` and `111` alternate with period
    2. A repression delayed by `delay` steps passes through that many relay nodes, `<repressor>_1` and so on, which
    copy the repressor one step later and start at its initial value. The oscillation remains, with period
    `6 (delay + 1)`.

    Args:
        x:
            The initial values of LacI, TetR, and cI.

        delay:
            The number of steps by which every repression is delayed.

    Returns:
        system:
            The system, with the values `LacI.x`, `TetR.x`, `cI.x`, and the relays `<repressor>_<k>.x`.

    Raises:
        ValidationError:
            The delay is negative.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import boolean_repressilator
        >>> [node.label for node in boolean_repressilator(delay=1).elements]
        ['LacI', 'TetR', 'cI', 'LacI_1', 'TetR_1', 'cI_1']
    """
    if delay < 0:
        error_msg = f'The delay must be a nonnegative number of steps, not {delay}.'
        raise ValidationError(error_msg)
    genes = [BooleanNodeElement(name, x=int(value)) for name, value in zip(REPRESSILATOR_GENES, x, strict=True)]
    relays = [BooleanNodeElement(f'{gene.label}_{step}', x=gene.x) for step in range(1, delay + 1) for gene in genes]
    functions: list[BooleanFunctionInteraction] = []
    for repressor, target in zip(REPRESSILATOR_GENES, (*REPRESSILATOR_GENES[1:], REPRESSILATOR_GENES[0]), strict=True):
        chain = [repressor, *(f'{repressor}_{step}' for step in range(1, delay + 1))]
        functions.extend(
            BooleanFunctionInteraction(f'relay_{relay}', target=relay, inputs=[source], table=(0, 1))
            for source, relay in itertools.pairwise(chain)
        )
        functions.append(
            BooleanFunctionInteraction(f'repression_{target}', target=target, inputs=[chain[-1]], table=(1, 0)),
        )
    return System(elements=[*genes, *relays], interactions=functions)


@register_system(
    field='biology/systems_biology',
    tags=('boolean network', 'noise', 'phase transition', 'mean field', 'stochastic map'),
    references=('Huepe and Aldana-González 2002', 'Gros 2015'),
    defaults={'until': 200, 'dt': 1},
)
def noisy_boolean_network(n: int = 1000, *, k: int = 3, eta: float = 0.1, seed: int | None = 0) -> System:
    """Return the Boolean network of Huepe and Aldana-González, nodes that follow the noisy majority of `k` inputs.

    Every node takes the sign of the sum of its `k` random inputs, flipped with the probability `eta`. The order
    parameter `Psi`, the time average of the absolute mean state, undergoes a continuous phase transition at the noise
    `eta_c = (1 - 1 / Phi'(0)) / 2` of mean-field theory, with `Phi'(0) = k C(k - 1, (k - 1) / 2) / 2^(k - 1)`, which is
    1/6 for `k = 3`. Below it the network orders, above it the mean state falls to fluctuations of order `1 /
    sqrt(n)`.

    Args:
        n:
            The number of nodes.

        k:
            The number of inputs of every node, odd.

        eta:
            The noise, the probability of a flip.

        seed:
            The seed of the inputs.

    Returns:
        system:
            The system, with the states `nodes.sigma`.

    Raises:
        ValidationError:
            The number of inputs is even.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import noisy_boolean_network
        >>> noisy_boolean_network(10).elements[0].prepare().size_
        10
    """
    if k % 2 == 0:
        error_msg = f'The number of inputs must be odd so that the majority is defined, not {k}.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(seed)
    inputs = np.zeros((n, n))
    for node in range(n):
        inputs[node, rng.choice(np.delete(np.arange(n), node), size=k, replace=False)] = 1.0
    nodes = GroupElement(
        'nodes',
        element=MajorityNodeElement,
        size=n,
        values={'sigma': np.ones(n), 'eta': eta},
    )
    majority = NoisyMajorityInteraction('majority', nodes='nodes', inputs=scipy.sparse.csr_array(inputs))
    return System(elements=[nodes], interactions=[majority], time=DiscreteTime())
