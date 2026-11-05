"""Define optimization interactions, which move a population of candidates one generation per step with pymoo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination

from ...errors import ValidationError
from ...specs import (
    CovarianceAdaptationSpec,
    DifferentialEvolutionSpec,
    EvolutionStrategySpec,
    ParticleSwarmSpec,
    RoleSpec,
)
from ...types import Array, ElementKind
from ..laws import BaseLaw, RuleLaw
from ..types import ModelViewProtocol
from ._interaction import Interaction

type AlgorithmSpec = DifferentialEvolutionSpec | ParticleSwarmSpec | EvolutionStrategySpec | CovarianceAdaptationSpec

ALGORITHMS = (DifferentialEvolutionSpec, ParticleSwarmSpec, EvolutionStrategySpec, CovarianceAdaptationSpec)
SEEDS = 2**31


class OptimizationInteraction(Interaction):
    """Move a group of candidates on an objective, one generation of a pymoo algorithm per step of discrete time.

    Every candidate holds a vector state in a box, from which the first generation starts. Every step asks the algorithm
    for candidates, evaluates the objective on their array, tells the algorithm the values, and writes the candidates of
    the generation into the members. The interaction reports `best`, the lowest value so far, and `mean`, the mean value
    of the generation, every step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of candidates.

        state:
            The name of the vector state of the candidates.

        objective:
            The function of the array of candidates, one row per candidate, that returns their values.

        bounds:
            The lower and the upper bounds of the box, each a number or one number per component.

        algorithm:
            The settings of the algorithm, which select it.

        phase:
            The name of the phase of the step the rule runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The algorithm is not an algorithm spec, or the objective is not callable, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.interactions import OptimizationInteraction
        >>> objective = lambda values: (values**2).sum(axis=1)
        >>> search = OptimizationInteraction('search', target='swarm', state='x', objective=objective, bounds=(-1, 1))
        >>> search.label
        'search'
    """

    reports: ClassVar[tuple[str, ...]] = ('best', 'mean')

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP), states=('state',))],
        state: str,
        objective: Callable[[Array], npt.ArrayLike],
        bounds: tuple[npt.ArrayLike, npt.ArrayLike],
        algorithm: AlgorithmSpec = DifferentialEvolutionSpec(),
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.objective = objective
        self.bounds = bounds
        self.algorithm = algorithm
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for an algorithm that is not a spec, or an objective that is not callable.

        Raises:
            ValidationError:
                The algorithm is not an algorithm spec, or the objective is not callable.
        """
        super()._check()
        if not isinstance(self.algorithm, ALGORITHMS):
            names = ', '.join(item.__name__ for item in ALGORITHMS)
            error_msg = f'The algorithm {self.algorithm!r} of {self.label!r} is no algorithm spec. Use one of {names}.'
            raise ValidationError(error_msg)
        if not callable(self.objective):
            error_msg = f'The objective of {self.label!r} must be a function of the array of candidates.'
            raise ValidationError(error_msg)

    def _differential_evolution(self, spec: DifferentialEvolutionSpec, candidates: Array) -> Algorithm:
        """Return differential evolution from the candidates."""
        return DE(
            pop_size=len(candidates),
            sampling=candidates,
            variant=spec.variant,
            F=spec.weight,
            CR=spec.crossover,
        )

    def _particle_swarm(self, spec: ParticleSwarmSpec, candidates: Array) -> Algorithm:
        """Return particle swarm optimisation from the candidates."""
        return PSO(
            pop_size=len(candidates),
            sampling=candidates,
            w=spec.inertia,
            c1=spec.cognitive,
            c2=spec.social,
            adaptive=False,
        )

    def _evolution_strategy(self, spec: EvolutionStrategySpec, candidates: Array) -> Algorithm:
        """Return an evolution strategy from the candidates."""
        return ES(pop_size=len(candidates), n_offsprings=spec.offspring, sampling=candidates)

    def _covariance_adaptation(self, spec: CovarianceAdaptationSpec, candidates: Array) -> Algorithm:
        """Return covariance matrix adaptation from the mean of the candidates."""
        return CMAES(x0=candidates.mean(axis=0), sigma=spec.sigma, pop_size=len(candidates))

    def _get_algorithm(self, candidates: Array, seed: int) -> Algorithm:
        """Return the algorithm of the spec, set up on the box of the candidates with a seed."""
        makers: dict[type, Callable[..., Algorithm]] = {
            DifferentialEvolutionSpec: self._differential_evolution,
            ParticleSwarmSpec: self._particle_swarm,
            EvolutionStrategySpec: self._evolution_strategy,
            CovarianceAdaptationSpec: self._covariance_adaptation,
        }
        size = candidates.shape[1]
        lower, upper = (np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in self.bounds)
        problem = Problem(n_var=size, n_obj=1, xl=lower, xu=upper)
        algorithm = makers[type(self.algorithm)](self.algorithm, candidates)
        algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
        return algorithm

    @staticmethod
    def _members_of(generation: Array, size: int) -> Array:
        """Return one row per member from the candidates of a generation, repeating them when they are fewer."""
        return generation[np.arange(size) % len(generation)]

    def _write(self, agents: Sequence[object], candidates: Array) -> None:
        """Write the candidates into the members."""
        for agent, row in zip(agents, self._members_of(candidates, len(agents)), strict=True):
            setattr(agent, self.state, row)

    def _advance(self, model: ModelViewProtocol) -> None:
        """Advance the population by one generation, unless the algorithm has converged, and report its values."""
        agents = model.agents(self._get_role('target'))
        if 'algorithm' not in model.storage:
            candidates = np.array([getattr(agent, self.state) for agent in agents], dtype=float)
            seed = int(model.rng.integers(SEEDS))
            model.storage.update(algorithm=self._get_algorithm(candidates, seed), best=np.inf)
        algorithm = cast(Algorithm, model.storage['algorithm'])
        if not algorithm.has_next():
            model.report('best', cast(float, model.storage['best']))
            model.report('mean', cast(float, model.storage['mean']))
            return
        population = algorithm.ask()
        generation = population.get('X')
        values = np.asarray(self.objective(generation), dtype=float).ravel()
        population.set('F', values[:, None])
        algorithm.tell(infills=population)
        best = min(cast(float, model.storage['best']), float(values.min()))
        model.storage.update(best=best, mean=float(values.mean()))
        self._write(agents, generation)
        model.report('best', best)
        model.report('mean', float(values.mean()))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one generation.

        Args:
            namespace:
                The namespace of the symbols the laws can use, which the rule does not read.

        Returns:
            laws:
                The rule that advances the population.
        """
        return [RuleLaw(self._advance, phase=self.phase)]
