"""Test the `_elementary_modes` module of `skcomplex.core.analysis`."""

from fractions import Fraction
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import nnls

from skcomplex.core.analysis import elementary_modes
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import LinearProgramLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import System
from skcomplex.core.errors import AnalysisError, EnumerationLimitError
from skcomplex.core.specs import LinearObjectiveSpec, ParameterSpec, StateSpec

Flux = Annotated[float, StateSpec(algebraic=True)]
NAMES = ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'b1', 'b2', 'vbio')
REVERSIBLE = ('v4', 'v6', 'b2')
MATRIX = np.array(
    [
        [-1, -1, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, -1, -1, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, -2],
        [1, 0, 1, 0, 0, 0, -1, 0, 0, -10],
    ],
    dtype=float,
)


class ToyElement(Element):
    """Toy network of Table 8.2 of Raman (2021), with a switch on `v1` and a right side of the balance of `A`."""

    def __init__(
        self,
        label,
        *,
        v1: Flux = 0.0,
        v2: Flux = 0.0,
        v3: Flux = 0.0,
        v4: Flux = 0.0,
        v5: Flux = 0.0,
        v6: Flux = 0.0,
        v7: Flux = 0.0,
        b1: Flux = 0.0,
        b2: Flux = 0.0,
        vbio: Flux = 0.0,
        switch: Annotated[float, ParameterSpec()] = 1.0,
        side: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.v1, self.v2, self.v3, self.v4, self.v5 = v1, v2, v3, v4, v5
        self.v6, self.v7, self.b1, self.b2, self.vbio = v6, v7, b1, b2, vbio
        self.switch, self.side = switch, side

    def _get_laws(self, s):
        variables = [getattr(s, name) for name in NAMES]
        balances = [
            sum(float(coefficient) * item for coefficient, item in zip(row, variables, strict=True)) == 0
            for row in MATRIX[1:]
        ]
        first = sum(float(coefficient) * item for coefficient, item in zip(MATRIX[0], variables, strict=True))
        bounds = [item <= 1000 for item in variables[1:]]
        bounds += [s.v1 <= 1000 * s.switch]
        bounds += [getattr(s, name) >= (-1000 if name in REVERSIBLE else 0) for name in NAMES]
        constraints = [first == s.side, *balances, *bounds]
        return [LinearProgramLaw(variables, constraints, [LinearObjectiveSpec(s.vbio)])]


def _toy(**values):
    """Return the system of the toy network."""
    return System(elements=[ToyElement('net', **values)], time=StaticTime())


def _column(analysis, name):
    """Return the column of a variable of the modes."""
    return analysis.modes[:, list(analysis.labels).index(f'net.{name}')]


def test_run_finds_the_two_modes_of_the_toy_network():
    """Test the 2 modes of the toy network, normalised to the uptake, each with its exact balances."""
    analysis = elementary_modes(_toy(), normalize='net.b1')
    assert analysis.modes.shape[0] == 2
    assert set(_column(analysis, 'vbio')) == {Fraction(1, 12)}
    assert set(_column(analysis, 'v6')) == {Fraction(5, 3)}
    assert {(item[0], item[1]) for item in zip(_column(analysis, 'v1'), _column(analysis, 'v2'), strict=True)} == {
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    }
    assert np.all(_column(analysis, 'v3') == _column(analysis, 'v2'))
    exact = [[Fraction(int(item)) for item in row] for row in MATRIX]
    for mode in analysis.modes:
        assert all(sum(a * b for a, b in zip(row, mode, strict=True)) == 0 for row in exact)


def test_run_decomposes_the_optimum_over_the_modes():
    """Test that the FBA optimum is a non-negative combination of the modes, with growth 1000 over their ratio 20."""
    analysis = elementary_modes(_toy(), normalize='net.b1')
    optimum = np.array([600, 0, 0, 550, 50, 1000, 100, 600, 1000, 50], dtype=float)
    modes = np.asarray(analysis.modes, dtype=float)
    _, residual = nnls(modes.T, optimum)
    assert residual < 1e-9
    ratios = _column(analysis, 'v6') / _column(analysis, 'vbio')
    assert set(ratios) == {Fraction(20)}
    assert 1000 / float(ratios[0]) == pytest.approx(50)


def test_run_finds_the_simple_cycles_of_a_circulation():
    """Test that the modes of the circulation of the complete digraph on 4 nodes are its 20 simple cycles."""
    graph = nx.complete_graph(4, create_using=nx.DiGraph)
    arcs = list(graph.edges)
    matrix = np.zeros((4, len(arcs)))
    for column, (tail, head) in enumerate(arcs):
        matrix[tail, column] -= 1
        matrix[head, column] += 1
    analysis = elementary_modes(matrix)
    found = {frozenset(np.flatnonzero(row).tolist()) for row in analysis.modes}
    cycles = {
        frozenset(arcs.index((cycle[index], cycle[(index + 1) % len(cycle)])) for index in range(len(cycle)))
        for cycle in nx.simple_cycles(graph)
    }
    assert len(found) == 20
    assert found == cycles


def test_prepare_removes_a_variable_that_a_switch_fixes_at_zero():
    """Test that with the switch of `v1` off only the mode through `v2` and `v3` remains."""
    analysis = elementary_modes(_toy(switch=0.0), normalize='net.b1')
    assert analysis.modes.shape[0] == 1
    assert 'net.v1' not in analysis.labels
    assert _column(analysis, 'v2')[0] == Fraction(1)
    assert _column(analysis, 'v3')[0] == Fraction(1)


def test_prepare_names_a_balance_with_a_right_side():
    """Test that a balance with the right side 5 raises an error that names it and says why."""
    with pytest.raises(AnalysisError, match=r'balance .*b1_net.*modes are defined at steady state'):
        elementary_modes(_toy(side=5.0))


def test_run_gives_the_same_vectors_as_pathways_and_modes():
    """Test that the pathways and the modes of the toy network hold the same 2 vectors."""
    modes = elementary_modes(_toy(), kind='modes', normalize='net.b1').modes
    pathways = elementary_modes(_toy(), kind='pathways', normalize='net.b1').modes
    assert {tuple(row) for row in modes} == {tuple(row) for row in pathways}


def test_run_keeps_the_lines_of_a_cone_that_is_not_pointed():
    """Test that a reversible internal cycle is a line of the pathways and two modes."""
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    modes = elementary_modes(matrix, irreversible=(), kind='modes')
    pathways = elementary_modes(matrix, irreversible=(), kind='pathways')
    assert modes.modes.shape[0] == 2
    assert pathways.modes.shape[0] == 0
    assert pathways.lines.tolist() == [[Fraction(1), Fraction(1)]]


def test_check_rejects_an_unknown_kind():
    """Test that an unknown kind raises an error that names the kinds."""
    with pytest.raises(AnalysisError, match='modes, pathways'):
        elementary_modes(MATRIX, kind='rays')


def test_prepare_rejects_a_coefficient_far_from_a_fraction():
    """Test that a coefficient that a fraction moves too far raises an error that names the variable."""
    matrix = np.array([[np.pi * 1e-7, -1.0]])
    with pytest.raises(AnalysisError, match='v0'):
        elementary_modes(matrix)


def test_run_stops_at_the_time_limit():
    """Test that a large cone raises the limit error that names the variables and the limit."""
    graph = nx.complete_graph(14, create_using=nx.DiGraph)
    arcs = list(graph.edges)
    matrix = np.zeros((14, len(arcs)))
    for column, (tail, head) in enumerate(arcs):
        matrix[tail, column] -= 1
        matrix[head, column] += 1
    with pytest.raises(EnumerationLimitError, match=f'{len(arcs)} variables.*0.5 s'):
        elementary_modes(matrix, timeout=0.5)


def test_run_finds_the_modes_and_pathways_of_figure_9_2_of_raman():
    """Test the 7 modes and 4 pathways of Raman's Figure 9.2, with its yields of G on C."""
    names = ('v1', 'v2', 'v3r', 'v4r', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11')
    rows = {
        'A': {'v1': 1, 'v2': -1},
        'B': {'v2': 1, 'v3r': -1, 'v5': -1},
        'C': {'v3r': 1, 'v4r': -1, 'v9': -1},
        'D': {'v5': 1, 'v6': -1, 'v8': -1},
        'E': {'v6': 1, 'v7': -1},
        'F': {'v9': 1, 'v10': -1},
        'G': {'v7': 2, 'v8': 2, 'v9': 1, 'v11': -1},
    }
    matrix = np.array([[float(row.get(name, 0)) for name in names] for row in rows.values()])
    irreversible = [position for position, name in enumerate(names) if not name.endswith('r')]
    settings = {'labels': names, 'irreversible': irreversible}
    modes = elementary_modes(matrix, kind='modes', **settings).modes
    pathways = elementary_modes(matrix, kind='pathways', **settings).modes
    assert modes.shape[0] == 7
    assert pathways.shape[0] == 4
    assert {tuple(row) for row in pathways} <= {tuple(row) for row in modes}
    uptake, product = names.index('v4r'), names.index('v11')
    yields = sorted({row[product] / -row[uptake] for row in pathways if row[uptake] < 0})
    assert yields == [Fraction(1), Fraction(2)]


def test_run_orders_the_rows_by_a_variable():
    """Test that the rows follow a variable, largest first."""
    matrix = np.array([[1.0, -1.0, -2.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0, -1.0]])
    analysis = elementary_modes(matrix, normalize='v0', sort='v4')
    assert analysis.modes[:, 4].tolist() == sorted(analysis.modes[:, 4].tolist(), reverse=True)
    assert analysis.modes[0, 4] == Fraction(1, 2)
