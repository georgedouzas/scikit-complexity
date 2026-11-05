"""Random wiring of technologies of the pool into a circuit, kept if it computes a wanted function more cheaply."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ._technology_view import TechnologyView


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
        >>> from skcomplex.systems.economics.innovation import InventionInteraction
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

    def _wire(self, pool: list[TechnologyView], system: SystemView) -> tuple[int, int, int]:
        """Return the arity, the truth table, and the cost of a random circuit of technologies of the pool."""
        rng = system.rng
        n_inputs = int(rng.integers(1, int(system.parameter('max_inputs')) + 1))
        n_parts = int(rng.integers(2, int(system.parameter('max_components')) + 1))
        signals, cost = InventionInteraction._input_signals(n_inputs), 0
        for _ in range(n_parts):
            part = pool[int(rng.integers(len(pool)))]
            pins = [signals[int(rng.integers(len(signals)))] for _ in range(int(part.arity))]
            signals.append(InventionInteraction._apply(int(part.table), int(part.arity), pins, n_inputs))
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

    def invent(self, system: SystemView) -> None:
        """Wire a random circuit and add it to the pool if it computes a wanted function better.

        Args:
            system:
                The system.
        """
        pool = system.agents('technologies', TechnologyView)
        arity, table, cost = self._wire(pool, system)
        if (arity, table) not in GOALS:
            return
        known = [technology for technology in pool if (technology.arity, technology.table) == (arity, table)]
        if not known:
            system.create('technologies', arity=arity, table=table, cost=cost)
        elif cost < known[0].cost:
            known[0].cost = cost

    @staticmethod
    def _input_signals(n_inputs: int) -> list[int]:
        """Return the truth tables of the primary inputs of a circuit."""
        rows = range(2**n_inputs)
        return [sum(1 << row for row in rows if row >> pin & 1) for pin in range(n_inputs)]

    @staticmethod
    def _apply(table: int, arity: int, signals: Sequence[int], n_inputs: int) -> int:
        """Return the truth table of a gate with a table and an arity, wired to the signals of a circuit."""
        output = 0
        for row in range(2**n_inputs):
            index = sum((signal >> row & 1) << pin for pin, signal in enumerate(signals[:arity]))
            output |= (table >> index & 1) << row
        return output
