"""Solver of lattice dynamics that applies every local rule, drive, cascade, and extremal law once per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ...components.laws import BoundaryLaw, CascadeLaw, DriveLaw, ExtremalLaw, LocalRuleLaw, UpdateLaw
from ...components.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ...specs import CascadeRecordSpec, RunStateSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ._solver import Solver


class LatticeSolver(Solver):
    """Solver of lattice dynamics that applies every local rule, drive, cascade, and extremal law once per step.

    Steps apply in the order the laws are declared, and the boundary laws apply after every step. Cascades record
    their size, duration, and area as events, and extremal laws record the site and the value of every application.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> from skcomplex.core.engine.solvers import LatticeSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> problem = PassCompiler().prepare(ElementaryAutomatonSystem(30, size=11).prepare().system_).run().problem_
        >>> stepper = LatticeSolver(random_state=0).prepare(problem)
        >>> step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> stepper.run(step).state_.astype(int).tolist()
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={LatticeSpace, NetworkSpace},
            laws={LocalRuleLaw, DriveLaw, CascadeLaw, ExtremalLaw, UpdateLaw, BoundaryLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.LATTICE

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Apply the lattice laws once per integer time step from `t` up to `until`.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector after the last whole step.
        """
        state = x.copy()
        for step in range(round(t), math.floor(until + 1e-9)):
            current = RunStateSpec(float(step + 1), state, p, self.rng_)
            for item in self.problem_.lattice:
                result = item.run(current).values_
                if isinstance(result, CascadeRecordSpec):
                    self.locator_.records.append(result)
                elif isinstance(result, list):
                    self.locator_.records.extend(result)
            state = self._bounded(float(step + 1), state, p)
        return state
