"""Name the solver that runs a problem of each form when no solver is passed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ..problems import Form
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
from ._rewriting import RewritingSolver
from ._scipy_ode import ScipyODESolver

DEFAULT_SOLVERS: dict[Form, type[BaseSolver]] = {
    Form.ODE: ScipyODESolver,
    Form.SDE: EulerMaruyamaSolver,
    Form.JUMP: GillespieSolver,
    Form.RENEWAL: RenewalSolver,
    Form.DISCRETE: IterationSolver,
    Form.FRACTIONAL: FractionalABMSolver,
    Form.LATTICE: LatticeSolver,
    Form.MONTE_CARLO: MonteCarloSolver,
    Form.AGENT: MesaSolver,
    Form.QUANTUM: LindbladSolver,
    Form.REWRITING: RewritingSolver,
}
