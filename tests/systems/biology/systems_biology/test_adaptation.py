"""Test the `_adaptation` module of `skcomplex.systems.biology.systems_biology`."""

import itertools

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.systems_biology import EnzymeAdaptationCircuitSystem


def _sensitivity_and_precision(links=None) -> tuple[float, float]:
    """Return the sensitivity and the precision of the output to the input step of 20 percent, as Ma et al.

    define.
    """
    system = (
        EnzymeAdaptationCircuitSystem().prepare().system_
        if links is None
        else EnzymeAdaptationCircuitSystem(links).prepare().system_
    )
    trajectory = simulate(system, dt=0.1, until=800)
    output, times = trajectory.get_values('C.X'), trajectory.times
    before, after = output[times < 300][-1], output[times >= 300]
    peak = after[np.argmax(np.abs(after - before))]
    sensitivity = abs(peak - before) / before / 0.2
    precision = 0.2 * before / max(abs(output[-1] - before), 1e-12)
    return sensitivity, precision


def test_negative_feedback_with_a_buffer_adapts():
    """Test that the default loop responds sensitively, `S > 1`, and returns precisely, `P > 10`."""
    sensitivity, precision = _sensitivity_and_precision()
    assert sensitivity > 1
    assert precision > 10


def test_a_cascade_without_feedback_does_not_adapt():
    """Test that the cascade A to C, with the buffer cut out of the loop, fails the criteria of adaptation."""
    sensitivity, precision = _sensitivity_and_precision(((0, 0, 0), (0, 0, 1), (1, 0, 0)))
    assert not (sensitivity > 1 and precision > 10)


def test_three_node_topologies_whose_input_reaches_the_output():
    """Test the count of Ma et al.: 16,038 of the 3^9 sign matrices route the input at A to the output C."""
    count = 0
    for signs in itertools.product((-1, 0, 1), repeat=9):
        matrix = np.reshape(signs, (3, 3))
        graph = nx.DiGraph((source, target) for target, source in zip(*np.nonzero(matrix), strict=True))
        count += graph.has_node(0) and graph.has_node(2) and nx.has_path(graph, 0, 2)
    assert count == 16038


def test_invalid_links_raise():
    """Test that a sign other than 1, -1, and 0 raises an error."""
    with pytest.raises(ValidationError):
        EnzymeAdaptationCircuitSystem(((0, 2, 0), (0, 0, 1), (1, 0, 0))).prepare()
