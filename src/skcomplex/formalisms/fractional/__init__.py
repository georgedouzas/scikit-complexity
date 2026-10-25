"""Fractional dynamics: states of fractional order, Caputo schemes, and fractional noise."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._laws import FractionalNoiseLaw
from ._noise import FractionalEulerMaruyamaSolver, fractional_gaussian_noise
from ._passes import CheckFractionalNoisePass, FractionalFormPass, FractionalProblemPass, order_issues
from ._solvers import FractionalABMSolver, GrunwaldLetnikovSolver, L1Solver

__all__: list[str] = [
    'FractionalABMSolver',
    'FractionalEulerMaruyamaSolver',
    'FractionalNoiseLaw',
    'GrunwaldLetnikovSolver',
    'L1Solver',
    'fractional_gaussian_noise',
]

register(
    Formalism(
        'fractional',
        laws=(FractionalNoiseLaw,),
        passes=(CheckFractionalNoisePass(), FractionalFormPass(), FractionalProblemPass()),
        forms=(Form.FRACTIONAL,),
        solvers={
            'grunwald-letnikov': GrunwaldLetnikovSolver,
            'l1': L1Solver,
            'fractional-abm': FractionalABMSolver,
            'fractional-euler-maruyama': FractionalEulerMaruyamaSolver,
        },
        defaults=(FractionalABMSolver,),
        checks=(order_issues,),
    ),
)
