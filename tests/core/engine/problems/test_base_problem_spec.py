"""Test the `BaseProblemSpec` class."""

import numpy as np

from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.problems import BaseProblemSpec, LayoutSpec
from skcomplex.core.enums import FormKind


def test_a_problem_answers_the_runtime_trivially_by_default():
    """A problem without the fields of a form answers the questions of the runtime trivially."""
    layout = LayoutSpec(())
    problem = BaseProblemSpec(FormKind.ODE, layout, layout, np.zeros(0), np.zeros(0), symbol('t'))
    assert (problem.stochastic, problem.adaptive_parameters, problem.get_lattice_shape('a.n')) == (False, {}, None)
