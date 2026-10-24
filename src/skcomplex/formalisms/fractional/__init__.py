"""Fractional dynamics: states of fractional order, Caputo schemes, and fractional noise."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._laws import FractionalNoise
from ._noise import FractionalEulerMaruyama, fractional_gaussian_noise
from ._passes import CheckFractionalNoisePass, FractionalFormPass, FractionalProblemPass, order_issues
from ._solvers import FractionalABM, GrunwaldLetnikov, L1Scheme

__all__: list[str] = [
    'FractionalABM',
    'FractionalEulerMaruyama',
    'FractionalNoise',
    'GrunwaldLetnikov',
    'L1Scheme',
    'fractional_gaussian_noise',
]

register(
    Formalism(
        'fractional',
        laws=(FractionalNoise,),
        passes=(CheckFractionalNoisePass(), FractionalFormPass(), FractionalProblemPass()),
        forms=(Form.FRACTIONAL,),
        solvers={
            'grunwald-letnikov': GrunwaldLetnikov,
            'l1': L1Scheme,
            'fractional-abm': FractionalABM,
            'fractional-euler-maruyama': FractionalEulerMaruyama,
        },
        defaults=(FractionalABM,),
        checks=(order_issues,),
    ),
)
