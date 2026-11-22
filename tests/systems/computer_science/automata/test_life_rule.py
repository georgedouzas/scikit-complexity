"""Test the `_life_rule` module of `skcomplex.systems.computer_science.automata`."""

import time

from skcomplex.systems.computer_science.automata import GameOfLifeSystem, LifeRuleInteraction


def test_life_rule_runs_a_large_board_in_real_time():
    """Test that a 1000 x 1000 Game of Life runs as one fused lookup at more than 30 steps per second."""
    system = GameOfLifeSystem((1000, 1000)).prepare()
    system.simulate(until=2, dt=1)
    start = time.perf_counter()
    system.simulate(until=90, dt=1)
    assert 90 / (time.perf_counter() - start) >= 30.0
    (rule,) = system.system_._problem.lattice
    assert (rule.lookup.kind, rule.fused) == ('outer', True)


def test_life_rule_shows_its_label_in_the_equations():
    """Test that the equations name the rule by the label of its interaction."""
    equations = GameOfLifeSystem((10, 10)).prepare().equations_
    assert equations == {'cells.alive[i]': "rule(neighbors(cells.alive, reduce='each'))"}


def test_life_rule_keeps_its_counts():
    """Test that the rule keeps the numbers of neighbours that bring cells to life and keep them alive."""
    rule = LifeRuleInteraction('rule', born=(3, 6), survive=(2, 3))
    assert (rule.born, rule.survive) == ((3, 6), (2, 3))
