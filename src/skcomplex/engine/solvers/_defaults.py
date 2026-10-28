"""Name the solver that runs a problem of each form when no solver is passed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import FormKind
from ._base import BaseSolver
from ._euler_maruyama import EulerMaruyamaSolver
from ._fractional_abm import FractionalABMSolver
from ._gillespie import GillespieSolver
from ._lattice import LatticeSolver
from ._lindblad import LindbladSolver
from ._maps import IterationSolver
from ._mesa import MesaSolver
from ._monte_carlo import MonteCarloSolver
from ._renewal import RenewalSolver
from ._scipy_ode import ScipyODESolver

DEFAULT_SOLVERS: dict[FormKind, type[BaseSolver]] = {
    FormKind.ODE: ScipyODESolver,
    FormKind.SDE: EulerMaruyamaSolver,
    FormKind.JUMP: GillespieSolver,
    FormKind.RENEWAL: RenewalSolver,
    FormKind.DISCRETE: IterationSolver,
    FormKind.FRACTIONAL: FractionalABMSolver,
    FormKind.LATTICE: LatticeSolver,
    FormKind.MONTE_CARLO: MonteCarloSolver,
    FormKind.AGENT: MesaSolver,
    FormKind.QUANTUM: LindbladSolver,
}
