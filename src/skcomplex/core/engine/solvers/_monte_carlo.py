"""Solver of Monte Carlo sampling that sweeps the lattice once per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws._sample import _SampleLaw
from ...components.spaces import LatticeSpace, NetworkSpace
from ...components.spaces._discrete_time import _DiscreteTime
from ...enums import FormKind, OrderKind
from ..backends import SupportSpec
from ._lattice import LatticeSolver


class MonteCarloSolver(LatticeSolver):
    """A solver of Monte Carlo sampling that sweeps a lattice once per unit of time.

    Every unit of time visits each site once and redraws its value by the sampling law of the lattice, such as a spin of
    a magnet that flips with the Boltzmann weight of its neighbours, so a cold magnet of 64 spins lines up within 50
    sweeps. It samples magnets, lattice gases, and the opinions of voters on a grid.

    Read more in the user guide, under Solvers.

    See Also:
        LatticeSolver: Applies the laws of a lattice in order.
        MetropolisSolver: Samples states by the Boltzmann weight of an objective.

    Examples:
        >>> from skcomplex.core.engine.solvers import MonteCarloSolver, StepSpec
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSystem
        >>> problem = IsingSystem(size=8, temperature=1.0).prepare().simulate(until=0).problem
        >>> step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 50.0)
        >>> bool(abs(MonteCarloSolver(random_state=0).prepare(problem).run(step).state_.mean()) > 0.9)
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_DiscreteTime},
            space={LatticeSpace, NetworkSpace},
            laws={_SampleLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.MONTE_CARLO
