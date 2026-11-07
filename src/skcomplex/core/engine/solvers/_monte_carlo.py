"""Solver of Monte Carlo sampling that sweeps the lattice once per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import SampleLaw
from ...components.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ...specs import SupportSpec
from ...types import FormKind, OrderKind
from ._lattice import LatticeSolver


class MonteCarloSolver(LatticeSolver):
    """Solver of Monte Carlo sampling that sweeps the lattice once per unit of time.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.compile import PassCompiler
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSystem
        >>> from skcomplex.core.engine.solvers import MonteCarloSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> problem = PassCompiler().prepare(IsingSystem(size=8, temperature=1.0).prepare().system_).run().problem_
        >>> stepper = MonteCarloSolver(random_state=0).prepare(problem)
        >>> state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 50.0)).state_
        >>> bool(abs(state.mean()) > 0.9)
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={LatticeSpace, NetworkSpace},
            laws={SampleLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.MONTE_CARLO
