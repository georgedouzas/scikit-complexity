"""Test the objective law and its compilation."""

import pytest

from skcomplex.core.base import LawKind, TimeKind, ValidationError, parameter, symbol
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.laws import Objective, Rate
from skcomplex.core.system import System

Pump = Element.define(
    'Pump',
    states={'x': 1.0},
    parameters={'u': parameter(0.0, adaptive=True, bounds=(0, 1)), 'weight': 2.0},
    laws=lambda s: [Rate(s.x, -s.x + s.u), Objective(s.x**2 + s.weight * s.u, terminal=s.x, sense='max', label='gain')],
)


def test_objective_declares_constraint_kind_and_times():
    """Test that an objective is a constraint law valid under static and continuous time."""
    assert Objective.law_kind is LawKind.OBJECTIVE
    assert Objective.requires.time == {TimeKind.STATIC, TimeKind.CONTINUOUS}


def test_objective_rejects_unknown_sense():
    """Test that an objective takes the sense `min` or `max` only."""
    with pytest.raises(ValidationError, match='Unknown sense'):
        Objective(symbol('x'), sense='best')


def test_objective_compiles_running_and_terminal_values():
    """Test that compiling keeps the running and terminal values with the sense, and leaves the form an ODE."""
    problem = compile_system(System(components=[Pump('pump')]))
    (objective,) = problem.objectives
    assert problem.form.name == 'ODE'
    assert objective.path == 'pump.gain'
    assert objective.sense == 'max'
    assert str(objective.terminal) == str(problem.symbols['pump.x'])
    assert set(problem.adaptive_parameters) == {'pump.u'}
    assert problem.adaptive_parameters['pump.u'].bounds == (0.0, 1.0)
