"""Define the solvers that move compiled problems forward in time.

A system picks the solver of the form its laws make, such as `ScipyODESolver` for the growth of a population or
`GillespieSolver` for the jumps of a few molecules, and takes another with `solver=`, such as
`LogisticGrowthSystem(solver=RK4Solver(dt=0.01))`.
"""

from ._base import BaseSolver
from ._caputo import CaputoSolver
from ._collocation import CollocationSolver
from ._cosimulation import CoSimulationSolver
from ._crank_nicolson import CrankNicolsonSolver
from ._etdrk4 import ETDRK4Solver
from ._euler import EulerSolver
from ._euler_maruyama import EulerMaruyamaSolver
from ._external import ExternalSolver
from ._fractional_abm import FractionalABMSolver
from ._fractional_euler_maruyama import FractionalEulerMaruyamaSolver
from ._gillespie import GillespieSolver
from ._gradient_descent import GradientDescentSolver
from ._gradient_flow import GradientFlowSolver
from ._grunwald_letnikov import GrunwaldLetnikovSolver
from ._heun import HeunSolver
from ._horizon_error import HorizonError
from ._imex2 import IMEX2Solver
from ._imex_euler import IMEXEulerSolver
from ._kinetics import KineticSolver
from ._l1 import L1Solver
from ._lammps import LammpsSolver
from ._lammps_fix_spec import LammpsFixSpec
from ._lattice import LatticeSolver
from ._maps import IterationSolver
from ._mesa import AgentView, MesaSolver, SystemView
from ._metropolis import MetropolisSolver
from ._milstein import MilsteinSolver
from ._minimize import MinimizeSolver
from ._monte_carlo import MonteCarloSolver
from ._optimal_control import OptimalControlSolver
from ._pontryagin import PontryaginSolver
from ._renewal import RenewalSolver
from ._rk4 import RK4Solver
from ._root import RootSolver
from ._scipy_ode import ScipyODESolver
from ._sde import SDESolver
from ._semi_lagrangian import SemiLagrangianSolver
from ._solver import Solver
from ._split import SplitSolver
from ._ssprk import SSPRKSolver
from ._step_spec import StepSpec
from ._strang_splitting import StrangSplittingSolver
from ._structure_state_spec import StructureStateSpec
from ._velocity_verlet import VelocityVerletSolver
from ._yoshida4 import Yoshida4Solver

__all__: list[str] = [
    'AgentView',
    'BaseSolver',
    'CaputoSolver',
    'CoSimulationSolver',
    'CollocationSolver',
    'CrankNicolsonSolver',
    'ETDRK4Solver',
    'EulerMaruyamaSolver',
    'EulerSolver',
    'ExternalSolver',
    'FractionalABMSolver',
    'FractionalEulerMaruyamaSolver',
    'GillespieSolver',
    'GradientDescentSolver',
    'GradientFlowSolver',
    'GrunwaldLetnikovSolver',
    'HeunSolver',
    'HorizonError',
    'IMEX2Solver',
    'IMEXEulerSolver',
    'IterationSolver',
    'KineticSolver',
    'L1Solver',
    'LammpsFixSpec',
    'LammpsSolver',
    'LatticeSolver',
    'MesaSolver',
    'MetropolisSolver',
    'MilsteinSolver',
    'MinimizeSolver',
    'MonteCarloSolver',
    'OptimalControlSolver',
    'PontryaginSolver',
    'RK4Solver',
    'RenewalSolver',
    'RootSolver',
    'SDESolver',
    'SSPRKSolver',
    'ScipyODESolver',
    'SemiLagrangianSolver',
    'Solver',
    'SplitSolver',
    'StepSpec',
    'StrangSplittingSolver',
    'StructureStateSpec',
    'SystemView',
    'VelocityVerletSolver',
    'Yoshida4Solver',
]
