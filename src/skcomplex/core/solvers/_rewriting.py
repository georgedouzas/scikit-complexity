"""Advance rewriting systems one generation per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..base import ComponentKind, DomainKind, Form, LawKind, SpaceKind, Supports, TimeKind
from ..compile import Problem, Structure, Vector
from ._base import Stepper, register_solver


@dataclass(frozen=True)
class RewritingState:
    """Structures of the rewriting elements at one generation, with their geometric measures.

    Args:
        structures:
            A mapping from element path to its word, patch, or host graph.

        reporters:
            A mapping from `<path>.length` and `<path>.area` to their values, for structures with a geometry.
    """

    structures: Mapping[str, Structure] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)


class Rewriter(Stepper):
    """Stepper of rewriting systems, which rewrites every structure in parallel once per generation.

    The structures live next to the state vector, whose components are their symbol, prototile, or label counts and
    sizes. Stochastic productions and the order of graph matches draw from the simulation's random stream.

    Examples:
        >>> from skcomplex.core.components import Word
        >>> from skcomplex.core.interactions import Rewrite
        >>> from skcomplex.core.laws import Production
        >>> from skcomplex.core.system import System
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.solvers import Rewriter
        >>> system = System(
        ...     components=[Word('algae', axiom='A')],
        ...     interactions=[Rewrite('rules', target='algae', rules=[Production('A', 'AB'), Production('B', 'A')])],
        ... )
        >>> trajectory = Simulation(system, solver=Rewriter()).run(until=5).trajectory_
        >>> trajectory['algae.size'].tolist()
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.DISCRETE},
        space=set(SpaceKind),
        laws={LawKind.REWRITE},
        domains={DomainKind.SEQUENCE, DomainKind.TILING, DomainKind.GRAPH, DomainKind.REAL, DomainKind.INTEGER},
        components={ComponentKind.ELEMENT, ComponentKind.GROUP},
    )

    form: ClassVar[Form] = Form.REWRITING

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the structures at generation 0.

        Args:
            problem:
                The compiled problem.

            rng:
                The random stream of stochastic productions and graph matches.
        """
        super().initialize(problem, rng)
        values = problem.initial_parameters
        self.structures_: dict[str, Any] = {item.path: item.initial(values) for item in problem.rewriting}

    def _state(self) -> Vector:
        """Return the state vector of the counts and sizes of the current structures."""
        counts = [value for item in self.problem_.rewriting for value in item.counts(self.structures_[item.path])]
        return np.asarray(counts, dtype=float)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
                item.path: item.rewrite(self.structures_[item.path], p, self.rng_) for item in self.problem_.rewriting
            }
        return self._state()

    def snapshot(self) -> RewritingState:
        """Return the current structures and their measures.

        Returns:
            snapshot:
                The structures, which later generations do not change, and their measures.
        """
        reporters: dict[str, float] = {}
        for path, structure in self.structures_.items():
            if structure.has_geometry:
                reporters[f'{path}.length'] = structure.length()
                reporters[f'{path}.area'] = structure.area()
        return RewritingState(dict(self.structures_), reporters)


register_solver('rewriter', Rewriter, default=True)
