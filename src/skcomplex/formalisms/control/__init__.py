"""Optimal control: solvers that optimise the objective laws of a system by collocation or Pontryagin's principle."""

from .._formalism import Formalism, register
from ._collocation import CollocationSolver
from ._optimal import ControlSolution
from ._pontryagin import PontryaginSolver

__all__: list[str] = [
    'CollocationSolver',
    'ControlSolution',
    'PontryaginSolver',
]

register(Formalism('control', solvers={'collocation': CollocationSolver, 'pontryagin': PontryaginSolver}))
