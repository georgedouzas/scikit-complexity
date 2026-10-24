"""Optimal control: solvers that optimise the objective laws of a system by collocation or Pontryagin's principle."""

from .._formalism import Formalism, register
from ._collocation import Collocation
from ._optimal import ControlSolution
from ._pontryagin import Pontryagin

__all__: list[str] = [
    'Collocation',
    'ControlSolution',
    'Pontryagin',
]

register(Formalism('control', solvers={'collocation': Collocation, 'pontryagin': Pontryagin}))
