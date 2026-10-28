"""Advance rewriting systems one generation per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np

from ...core.laws import GraphProductionLaw, ProductionLaw, SubstitutionLaw
from ...core.spaces import CayleySpace, ContinuousSpace, DiscreteTime, LatticeSpace, MeshSpace, NetworkSpace
from ...types import Array
from ..problems import Form, OrderKind, State, SupportSpec
from ..problems._rewriting_problem import RewritingProblem
from ._base import BaseSolver
from ._structure_state import StructureState


class RewritingSolver(BaseSolver[RewritingProblem]):
    """Solver of rewriting systems, which rewrites every structure in parallel once per generation.

    The structures live next to the state vector, whose components are their symbol, prototile, or label counts and
    sizes. Stochastic productions and the order of graph matches draw from the simulation's random stream.

    Examples:
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.core.laws import ProductionLaw
        >>> from skcomplex.engine.solvers import RewritingSolver
        >>> from skcomplex.core.elements import WordElement
        >>> class AlgaeElement(WordElement):
        ...     def _get_laws(self, namespace):
        ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> system = System(elements=[AlgaeElement('algae', axiom='A')], time=DiscreteTime())
        >>> trajectory = Simulation(system, solver=RewritingSolver()).check().run(until=5).trajectory_
        >>> trajectory.values('algae.size').tolist()
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={ProductionLaw, SubstitutionLaw, GraphProductionLaw},
            orders={OrderKind.INTEGER},
        )

    @classmethod
    def _get_form(cls) -> Form:
        """Return the form the solver solves."""
        return Form.REWRITING

    def _prepare(self, problem: RewritingProblem) -> None:
        """Prepare the structures at generation 0.

        Args:
            problem:
                The compiled problem.
        """
        super()._prepare(problem)
        self.systems_ = problem.rewriting if isinstance(problem, RewritingProblem) else ()
        self.structures_: dict[str, Any] = {item.path: item.prepare(problem).structure_ for item in self.systems_}
        self._collect()

    def _state(self) -> Array:
        """Return the state vector of the counts and sizes of the current structures."""
        counts = [value for item in self.systems_ for value in item.counts(self.structures_[item.path])]
        return np.asarray(counts, dtype=float)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Rewrite every structure once per generation from `t` to `until`.

        Args:
            t:
                The current generation.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The generation to advance to.

        Returns:
            state:
                The counts and sizes at generation `until`.
        """
        for generation in range(round(until - t)):
            state = State(t + generation, x, p, self.rng_)
            self.structures_ = {item.path: item.run(state) for item in self.systems_}
        return self._state()

    def _collect(self) -> None:
        """Set the snapshot of the current structures, which later generations do not change, and their measures."""
        reporters: dict[str, float] = {}
        for path, structure in self.structures_.items():
            if structure.has_geometry:
                reporters[f'{path}.length'] = structure.length()
                reporters[f'{path}.area'] = structure.area()
        self.events_ = []
        self.snapshot_ = StructureState(dict(self.structures_), reporters)
