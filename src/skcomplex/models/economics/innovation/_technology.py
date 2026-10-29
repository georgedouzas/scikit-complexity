"""Innovation models: technologies that improve by redesigning their components or combining logic circuits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model

NAND = (2, 0b0111)
FALSE = (0, 0)
TRUE = (0, 1)


def _truth_table(arity: int, function: Callable[..., bool]) -> int:
    """Return the truth table of a Boolean function, bit `r` holding its value on the input bits of row `r`."""
    rows = range(2**arity)
    return sum(int(function(*(bool(row >> pin & 1) for pin in range(arity)))) << row for row in rows)


GOALS = frozenset(
    (arity, _truth_table(arity, function))
    for arity, function in (
        (1, lambda a: not a),
        (2, lambda a, b: a and b),
        (2, lambda a, b: a or b),
        (2, lambda a, b: a != b),
        (2, lambda a, b: not (a or b)),
        (2, lambda a, b: a == b),
        (2, lambda a, b: (not a) or b),
        (3, lambda a, b, c: a and b and c),
        (3, lambda a, b, c: a or b or c),
        (3, lambda a, b, c: (a != b) != c),
        (3, lambda a, b, c: (a and b) or (a and c) or (b and c)),
    )
)


def _input_signals(n_inputs: int) -> list[int]:
    """Return the truth tables of the primary inputs of a circuit."""
    rows = range(2**n_inputs)
    return [sum(1 << row for row in rows if row >> pin & 1) for pin in range(n_inputs)]


def _apply(table: int, arity: int, signals: Sequence[int], n_inputs: int) -> int:
    """Return the truth table of a gate with a table and an arity, wired to signals of a circuit with given inputs."""
    output = 0
    for row in range(2**n_inputs):
        index = sum((signal >> row & 1) << pin for pin, signal in enumerate(signals[:arity]))
        output |= (table >> index & 1) << row
    return output


def _components(cost: Array, dependencies: Array, rng: np.random.Generator) -> tuple[Array, npt.NDArray[np.intp]]:
    """Return the proposed costs of a redesign of a random component and of the components that depend on it."""
    changed = np.flatnonzero(dependencies[:, int(rng.integers(len(cost)))])
    proposal = cost.copy()
    proposal[changed] = rng.random(len(changed))
    return proposal, changed


def _design_structure(n_components: int, d: int, rng: np.random.Generator) -> Array:
    """Return a design structure matrix where a change of every component changes it and `d - 1` other components."""
    matrix = np.eye(n_components)
    for column in range(n_components):
        others = rng.choice([row for row in range(n_components) if row != column], d - 1, replace=False)
        matrix[others, column] = 1.0
    return matrix


class PartElement(Element):
    """PartElement of a technology, described by its unit cost.

    Args:
        label:
            The label of the agent or group.

        location:
            The initial location, unused without a space.

        cost:
            The initial unit cost.

    Examples:
        >>> from skcomplex.models.economics.innovation import PartElement
        >>> GroupElement('components', element=PartElement, size=5, values={'cost': 0.5}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cost = cost

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since redesigns change several components at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class RedesignInteraction(Interaction):
    """Attempt to cut the cost of a technology by redesigning a random component and the components that depend on it.

    The redesigned components get new costs drawn uniformly from `[0, 1]`, and the redesign is kept only if the total
    cost falls. `dependencies[i][j]` is 1 when a change of component `j` requires a change of component `i`.

    Args:
        label:
            The label of the interaction.

        dependencies:
            The design structure matrix.

        **values:
            The values of the declared variables, by name:

            components:
                The path of the group of components.

    Examples:
        >>> from skcomplex.models.economics.innovation import RedesignInteraction
        >>> RedesignInteraction('redesign', components='components', dependencies=[[1.0]]).dependencies
        [[1.0]]
    """

    def __init__(
        self,
        label: str,
        *,
        components: Annotated[str, RoleSpec()],
        dependencies: Sequence[Sequence[float]] = (),
    ) -> None:
        super().__init__(label)
        self.components = components
        self.dependencies = dependencies

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the redesign rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.redesign)]

    def redesign(self, model: ModelView) -> None:
        """Propose new costs for a random component and its dependents and keep them if the total cost falls.

        Args:
            model:
                The model.
        """
        components = sorted(model.agents('components', AgentView), key=lambda component: component.id)
        cost = np.array([component.cost for component in components])
        proposal, changed = _components(cost, np.asarray(self.dependencies, dtype=float), model.rng)
        if proposal.sum() >= cost.sum():
            return
        for index in changed:
            components[index].cost = float(proposal[index])


class TechnologyElement(Element):
    """Logic circuit in the pool of technologies, described by the truth table of the function it computes.

    Bit `r` of the table is the output on the inputs whose bits form `r`, with the first input as the lowest bit.

    Args:
        label:
            The label of the agent or group.

        arity:
            The number of inputs.

        table:
            The truth table.

        cost:
            The number of NAND gates in the circuit.

    Examples:
        >>> from skcomplex.models.economics.innovation import TechnologyElement
        >>> GroupElement(
        ...     'technologies', element=TechnologyElement, size=3, values={'arity': [2, 0, 0], 'table': [7, 0, 1]}
        ... ).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        arity: Annotated[float, StateSpec(domain=NN)] = 2,
        table: Annotated[float, StateSpec(domain=NN)] = NAND[1],
        cost: Annotated[float, StateSpec(domain=NN)] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.arity = arity
        self.table = table
        self.cost = cost

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since inventions combine several technologies at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class InventionInteraction(Interaction):
    """Random wiring of technologies of the pool into a circuit, kept if it computes a wanted function more cheaply.

    A circuit has between one and `max_inputs` inputs and wires between two and `max_components` technologies in a
    row, each pin to an input or to the output of an earlier technology. The output of the last one is the circuit's
    output. A circuit that computes a wanted function joins the pool, or replaces the technology of that function when
    it uses fewer NAND gates.

    Args:
        label:
            The label of the interaction.

        technologies:
            The path of the group of technologies.

        max_components:
            The largest number of technologies in a circuit.

        max_inputs:
            The largest number of inputs of a circuit.

    Examples:
        >>> from skcomplex.models.economics.innovation import InventionInteraction
        >>> InventionInteraction('invention', technologies='technologies', max_components=6).max_components
        6
    """

    def __init__(
        self,
        label: str,
        *,
        technologies: Annotated[str, RoleSpec()],
        max_components: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12,
        max_inputs: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
    ) -> None:
        super().__init__(label)
        self.technologies = technologies
        self.max_components = max_components
        self.max_inputs = max_inputs

    def _wire(self, pool: list[AgentView], model: ModelView) -> tuple[int, int, int]:
        """Return the arity, the truth table, and the cost of a random circuit of technologies of the pool."""
        rng = model.rng
        n_inputs = int(rng.integers(1, int(model.parameter('max_inputs')) + 1))
        n_parts = int(rng.integers(2, int(model.parameter('max_components')) + 1))
        signals, cost = _input_signals(n_inputs), 0
        for _ in range(n_parts):
            part = pool[int(rng.integers(len(pool)))]
            pins = [signals[int(rng.integers(len(signals)))] for _ in range(int(part.arity))]
            signals.append(_apply(int(part.table), int(part.arity), pins, n_inputs))
            cost += int(part.cost)
        return n_inputs, signals[-1], cost

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the invention rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.invent)]

    def invent(self, model: ModelView) -> None:
        """Wire a random circuit and add it to the pool if it computes a wanted function better.

        Args:
            model:
                The model.
        """
        pool = model.agents('technologies', AgentView)
        arity, table, cost = self._wire(pool, model)
        if (arity, table) not in GOALS:
            return
        known = [technology for technology in pool if (technology.arity, technology.table) == (arity, table)]
        if not known:
            model.create('technologies', arity=arity, table=table, cost=cost)
        elif cost < known[0].cost:
            known[0].cost = cost


@register_model(
    field='economics/innovation',
    tags=('innovation', 'technology', 'learning curve', 'agents'),
    references=('Thurner et al. 2018', 'McNerney et al. 2011'),
    defaults={'until': 200},
    observables=lambda system: [ObservableSpec('total_cost', group='components', reduce='sum', variable='cost')],
)
def nk_technology(
    n_components: int = 10,
    d: int = 2,
    *,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return McNerney's model of technological improvement, whose cost falls as a power of the cumulative production.

    Every step is one unit of production and one attempt to redesign a random component together with the components
    that depend on it. Redesigns are kept only when they cut the total cost, which falls like Wright's law, more slowly
    for designs whose components depend on more others.

    Args:
        n_components:
            The number of components.

        d:
            The number of components that a redesign of one component changes, itself included.

        random_state:
            The seed or generator of the design structure matrix and the initial costs.

    Returns:
        system:
            The model, with the group `components` and the total cost `total_cost`.

    Examples:
        >>> from skcomplex.models.economics.innovation import nk_technology
        >>> nk_technology(5, 1).interactions[0].dependencies[0]
        [1.0, 0.0, 0.0, 0.0, 0.0]
    """
    rng = np.random.default_rng(random_state)
    dependencies = _design_structure(n_components, d, rng)
    components = GroupElement(
        'components',
        element=PartElement,
        size=n_components,
        values={'cost': rng.random(n_components)},
    )
    return System(
        elements=[components],
        interactions=[RedesignInteraction('redesign', components='components', dependencies=dependencies.tolist())],
    )


@register_model(
    field='economics/innovation',
    tags=('innovation', 'technology', 'combinatorial evolution', 'agents'),
    references=('Thurner et al. 2018', 'Arthur and Polak 2006'),
    defaults={'until': 200},
)
def arthur_polak(*, max_components: int = 12, max_inputs: int = 3) -> System:
    """Return Arthur and Polak's evolution of technology, which builds logic functions from NAND gates.

    The pool starts with the NAND gate and the constants 0 and 1. Every step wires a random circuit from the pool and
    keeps it when it computes a wanted function, negation, two-input and, or, exclusive or, nor, equivalence, and
    implication, or three-input and, or, parity, and majority, so simple inventions become the building blocks of
    complex ones.

    Args:
        max_components:
            The largest number of technologies in a circuit.

        max_inputs:
            The largest number of inputs of a circuit.

    Returns:
        system:
            The model, with the pool `technologies`, whose size is the reporter `technologies.count`.

    Examples:
        >>> from skcomplex.models.economics.innovation import arthur_polak
        >>> arthur_polak().elements[0].check().prepare().size_
        3
    """
    primitives = (NAND, FALSE, TRUE)
    technologies = GroupElement(
        'technologies',
        element=TechnologyElement,
        size=len(primitives),
        values={
            'arity': [float(arity) for arity, _ in primitives],
            'table': [float(table) for _, table in primitives],
            'cost': [1.0, 0.0, 0.0],
        },
        variable=True,
    )
    invention = InventionInteraction(
        'invention',
        technologies='technologies',
        max_components=float(max_components),
        max_inputs=float(max_inputs),
    )
    return System(elements=[technologies], interactions=[invention])
