"""Advance rewriting systems one generation per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Any, ClassVar

import numpy as np

from ...core.base import DomainKind, ElementKind, Form, LawKind, OrderKind, SpaceKind, Supports, TimeKind, Vector
from ...core.problems import BaseSolver
from ...engine.solvers import StructureState
from ._passes import RewritingProblem


class RewritingSolver(BaseSolver[RewritingProblem]):
    """Solver of rewriting systems, which rewrites every structure in parallel once per generation.

    The structures live next to the state vector, whose components are their symbol, prototile, or label counts and
    sizes. Stochastic productions and the order of graph matches draw from the simulation's random stream.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.formalisms.rewriting import ProductionLaw, RewriteLaw, RewritingSolver, WordElement
        >>> class AlgaeElement(WordElement):
        ...     def laws(self, s):
        ...         return [RewriteLaw([ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])]
        >>> system = System(components=[AlgaeElement('algae', axiom='A')])
        >>> trajectory = Simulation(system, solver=RewritingSolver()).run(until=5).trajectory_
        >>> trajectory.values('algae.size').tolist()
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.DISCRETE},
        space=set(SpaceKind.registered()),
        laws={LawKind.REWRITE},
        domains={DomainKind.SEQUENCE, DomainKind.TILING, DomainKind.GRAPH, DomainKind.REAL, DomainKind.INTEGER},
        components={ElementKind.ELEMENT, ElementKind.GROUP},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.REWRITING

    def _prepare(self, problem: RewritingProblem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the structures at generation 0.

        Args:
            problem:
                The compiled problem.

            rng:
                The random stream of stochastic productions and graph matches.

            until:
                The end time of the run.
        """
        super()._prepare(problem, rng, until)
        values = problem.initial_parameters
        self.systems_ = problem.rewriting if isinstance(problem, RewritingProblem) else ()
        self.structures_: dict[str, Any] = {item.path: item.initial(values) for item in self.systems_}
        self._collect()

    def _state(self) -> Vector:
        """Return the state vector of the counts and sizes of the current structures."""
        counts = [value for item in self.systems_ for value in item.counts(self.structures_[item.path])]
        return np.asarray(counts, dtype=float)

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
        del x
        for _ in range(round(until - t)):
            self.structures_ = {
                item.path: item.rewrite(self.structures_[item.path], p, self.rng_) for item in self.systems_
            }
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
