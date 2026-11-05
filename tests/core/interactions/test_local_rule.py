"""Test the `_local_rule` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import LocalRuleInteraction
from skcomplex.core.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, TableSpec, TotalisticSpec, VisualSpec

SIZE = 9


class CellElement(Element):
    """Cell with a value `c` and a mark `m`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec()] = 0.0,
        m: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.m = m


def _run(rule, until, values=None):
    """Return the trajectory of a local rule on a ring of cells with one live cell in the middle."""
    initial = np.zeros(SIZE)
    initial[SIZE // 2] = 1.0
    cells = LatticeElement('cells', element=CellElement, space=LatticeSpace(SIZE), values=values or {'c': initial})
    system = System(elements=[cells], interactions=[rule], time=DiscreteTime())
    return Simulation(system, random_state=0).prepare().run(until=until).trajectory_


def test_totalistic_rule_of_the_neighbours_draws_rule_90():
    """Test that the parity of the two neighbours grows the pattern of the elementary rule 90."""
    parity = TotalisticSpec({1: 1})
    rule = LocalRuleInteraction(
        'rule',
        target='cells',
        state='c',
        rule=parity,
        neighborhood=VonNeumannNeighborhood(n_dim=1),
    )
    values = _run(rule, 2).values('cells.c')
    assert values[1].tolist() == [0, 0, 0, 1, 0, 1, 0, 0, 0]
    assert values[2].tolist() == [0, 0, 1, 0, 0, 0, 1, 0, 0]


def test_table_rule_reads_the_ordered_neighbourhood():
    """Test that a table of the ordered neighbourhood moves a live cell one site to the right."""
    shift = TableSpec({(0, 1, 0): 1, (1, 0, 0): 0}, default=0)
    around = VonNeumannNeighborhood(n_dim=1, center=True)
    rule = LocalRuleInteraction('shift', target='cells', state='c', rule=shift, neighborhood=around)
    values = _run(rule, 1).values('cells.c')
    assert int(np.argmax(values[1])) == int(np.argmax(values[0])) + 1


def test_function_rule_sets_two_states_together():
    """Test that a function rule sets two states, the shifted value and a mark of the cells that changed."""
    around = VonNeumannNeighborhood(n_dim=1, center=True)

    def shift(*values):
        c, _, right, m = values[:4]
        return right, np.where(right != c, 1.0, m)

    rule = LocalRuleInteraction('shift', target='cells', state=('c', 'm'), rule=shift, neighborhood=around)
    trajectory = _run(rule, 1)
    assert int(np.argmax(trajectory.values('cells.c')[1])) == SIZE // 2 - 1
    assert trajectory.values('cells.m')[1].sum() == 2.0


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'state': 'not a name'}, 'Give state names'), ({'neighborhood': 'von_neumann'}, 'neighbourhood')],
)
def test_check_rejects_states_and_neighbourhoods_it_cannot_read(arguments, match):
    """Test that a state that is not a name or a neighbourhood that is not one raise errors."""
    defaults = {'state': 'c', 'rule': TotalisticSpec({1: 1}), 'neighborhood': VonNeumannNeighborhood(n_dim=1)}
    with pytest.raises(ValidationError, match=match):
        LocalRuleInteraction('rule', target='cells', **(defaults | arguments)).prepare()
