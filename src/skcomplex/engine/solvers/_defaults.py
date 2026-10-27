"""Name the solver that runs a problem of each form when no solver is passed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ..problems import Form
from ._base import BaseSolver
from ._fractional import FractionalABMSolver
from ._jumps import GillespieSolver, RenewalSolver
from ._lattices import LatticeSolver, MonteCarloSolver
from ._maps import IterationSolver
from ._mesa import MesaSolver
from ._ode import ScipyODESolver
from ._quantum import LindbladSolver
from ._rewriting import RewritingSolver
from ._sde import EulerMaruyamaSolver

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
