"""Solver of lattice dynamics that applies every local rule, drive, relaxation, and extremal law once per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import cast

from ...core.laws import BoundaryLaw, DriveLaw, ExtremalLaw, LocalRuleLaw, RelaxationLaw, UpdateLaw
from ...core.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ...specs import RunStateSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import BaseKernel
from ..problems._extremal_kernel import ExtremalKernel
from ..problems._relaxation_kernel import RelaxationKernel
from ._solver import Solver


class LatticeSolver(Solver):
    """Solver of lattice dynamics that applies every local rule, drive, relaxation, and extremal law once per step.

    Steps apply in the order the laws are declared, and the boundary laws apply after every step. Relaxations record
    their size, duration, and area as events, and extremal laws record the site and the value of every application.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.computer_science.automata import elementary_automaton
        >>> from skcomplex.engine.solvers import LatticeSolver
        >>> from skcomplex.specs import StepSpec
        >>> problem = compile_system(elementary_automaton(30, size=11))
        >>> stepper = LatticeSolver(random_state=0).check().prepare(problem)
        >>> stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).astype(int).tolist()
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={LatticeSpace, NetworkSpace},
            laws={LocalRuleLaw, DriveLaw, RelaxationLaw, ExtremalLaw, UpdateLaw, BoundaryLaw},
            orders={OrderKind.INTEGER},
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
            for item in cast('tuple[BaseKernel, ...]', self.problem_.lattice):
                result = item.run(current)
                if isinstance(item, RelaxationKernel) and result is not None:
                    self.locator_.records.append(result)
                elif isinstance(item, ExtremalKernel):
                    self.locator_.records.extend(result)
            state = self._bounded(float(step + 1), state, p)
        return state
