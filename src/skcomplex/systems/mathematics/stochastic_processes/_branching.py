"""Stochastic process systems of branching processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, CallSpec, UpdateLaw
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

OFFSPRING_LAWS = ('poisson', 'geometric')
PROBABILITY_TOLERANCE = 1e-9


class IndividualView(AgentViewProtocol, Protocol):
    """View of an individual of a branching process, with its values as attributes."""

    generation: float


class IndividualElement(Element):
    """Individuals of a branching process, each with its generation `generation`.

    Args:
        label:
            The label of the individuals.

        generation:
            The initial generation, one per individual or shared.

        size:
            The number of individuals, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import IndividualElement
        >>> IndividualElement('population', size=10).size
        10
    """

    visual = VisualSpec('agents', {'color_by': 'generation'})

    def __init__(
        self,
        label: str,
        *,
        generation: Annotated[object, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.generation = generation


class GenerationElement(Element):
    """The size of the current generation of a branching process, as one count.

    Args:
        label:
            The label of the generation.

        count:
            The initial number of individuals.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import GenerationElement
        >>> GenerationElement('population', count=10.0).count
        10.0
    """

    def __init__(self, label: str, *, count: Annotated[float, NN] = 1.0) -> None:
        super().__init__(label)
        self.count = count


class BranchingInteraction(Interaction):
    """Replacement of every individual by a random number of offspring at every step.

    The offspring belong to the next generation and act from the next step on, so every step is one generation. The
    number of offspring is Poisson or geometric of mean `m`, or drawn from the probabilities `p_k` of `k` offspring.

    Args:
        label:
            The label of the interaction.

        target:
            The individuals that reproduce and die.

        m:
            The mean number of offspring of an individual for a Poisson or geometric law.

        offspring:
            The law of the number of offspring, `'poisson'`, `'geometric'`, or the probabilities of 0, 1, 2, and more
            offspring, which sum to 1.

    Raises:
        ValidationError:
            The offspring law is unknown, or its probabilities are negative or do not sum to 1.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import BranchingInteraction
        >>> BranchingInteraction('reproduction', m=1.5, offspring='geometric').offspring
        'geometric'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'population',
        m: float = 1.0,
        offspring: str | Sequence[float] = 'poisson',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.m = m
        self.offspring = offspring

    @staticmethod
    def _check_offspring(offspring: str | Sequence[float]) -> None:
        """Raise an error for an unknown offspring law or invalid probabilities."""
        if isinstance(offspring, str):
            if offspring not in OFFSPRING_LAWS:
                error_msg = (
                    f'Unknown offspring law {offspring!r}. Use one of {", ".join(OFFSPRING_LAWS)}, '
                    'or a sequence of probabilities.'
                )
                raise ValidationError(error_msg)
            return
        probabilities = np.asarray(offspring, dtype=float)
        if probabilities.ndim != 1 or (probabilities < 0).any() or abs(probabilities.sum() - 1) > PROBABILITY_TOLERANCE:
            error_msg = f'The offspring probabilities {list(offspring)} must be nonnegative and sum to 1.'
            raise ValidationError(error_msg)

    @staticmethod
    def _draw_offspring(offspring: str | Sequence[float], m: float, count: int, rng: np.random.Generator) -> int:
        """Return the total number of offspring of `count` individuals.

        The total is a Poisson draw of mean `count m`, a negative binomial draw for geometric offspring, or the sum of
        a multinomial draw over the numbers of offspring of a finite law.
        """
        if count == 0:
            return 0
        if offspring == 'poisson':
            return int(rng.poisson(count * m))
        if offspring == 'geometric':
            return int(rng.negative_binomial(count, 1 / (1 + m)))
        probabilities = np.asarray(offspring, dtype=float)
        return int(rng.multinomial(count, probabilities / probabilities.sum()) @ np.arange(len(probabilities)))

    def _check_arguments(self) -> None:
        """Raise an error for an offspring law the interaction cannot take.

        Raises:
            ValidationError:
                The offspring law is unknown, or its probabilities are negative or do not sum to 1.
        """
        super()._check_arguments()
        self._check_offspring(self.offspring)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the reproduction of every individual as the rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._replace)]

    def _replace(self, agent: IndividualView, system: SystemView) -> None:
        """Create the offspring of an individual and remove it."""
        for _ in range(self._draw_offspring(self.offspring, self.m, 1, system.rng)):
            agent.create(generation=agent.generation + 1)
        agent.remove()


class GenerationInteraction(Interaction):
    """Replacement of the size of a generation by the total offspring of its members.

    The total is drawn at once, as a Poisson draw of mean `n m`, a negative binomial draw for geometric offspring, or
    the sum of a multinomial draw over the numbers of offspring of a finite law, so every step is one generation.

    Args:
        label:
            The label of the interaction.

        target:
            The size of the generation.

        m:
            The mean number of offspring of an individual for a Poisson or geometric law.

        offspring:
            The law of the number of offspring, `'poisson'`, `'geometric'`, or the probabilities of 0, 1, 2, and more
            offspring, which sum to 1.

    Raises:
        ValidationError:
            The offspring law is unknown, or its probabilities are negative or do not sum to 1.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import GenerationInteraction
        >>> GenerationInteraction('reproduction', offspring=[0.25, 0.5, 0.25]).offspring
        [0.25, 0.5, 0.25]
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'population',
        m: float = 1.0,
        offspring: str | Sequence[float] = 'poisson',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.m = m
        self.offspring = offspring

    def _check_arguments(self) -> None:
        """Raise an error for an offspring law the interaction cannot take.

        Raises:
            ValidationError:
                The offspring law is unknown, or its probabilities are negative or do not sum to 1.
        """
        super()._check_arguments()
        BranchingInteraction._check_offspring(self.offspring)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the draw of the size of the next generation as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        count = context.target.count
        self.laws_: list[BaseLaw] = [UpdateLaw(count, CallSpec(self._draw_total, (count,), random=True))]

    def _draw_total(self, t: float, count: float, *, rng: np.random.Generator) -> float:
        """Return the total number of offspring of `count` individuals."""
        return float(BranchingInteraction._draw_offspring(self.offspring, self.m, int(count), rng))


class GaltonWatsonSystem(BaseSystem):
    """A Galton-Watson branching process with Poisson, geometric, or finite offspring.

    Every step is one generation.

    The mean size of generation `n` is `n_individuals m^n`, where `m` is the mean number of offspring. The process dies
    out with certainty for `m <= 1`, unless every individual has exactly one offspring, and survives with a positive
    probability for `m > 1`. With `individuals`, the system has the group `population`, whose count is the size of the
    current generation. Without it, the system has the state `population.count` alone, whose next value is drawn as
    the total offspring of the generation, so a large generation costs no more than a small one. The system draws
    nothing when it is prepared, so the `random_state` of `simulate` seeds its runs.

    Args:
        n_individuals:
            The size of the first generation.

        m:
            The mean number of offspring of an individual for a Poisson or geometric law.

        offspring:
            The law of the number of offspring, `'poisson'`, `'geometric'`, or the probabilities of 0, 1, 2, and more
            offspring, which sum to 1.

        individuals:
            Whether every individual is an agent of the group `population`, or only the size of the generation is
            tracked.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The offspring law is unknown, or its probabilities are negative or do not sum to 1.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import GaltonWatsonSystem
        >>> GaltonWatsonSystem(5, m=1.2).prepare().system_.elements[0].size
        5
        >>> system = GaltonWatsonSystem(5, offspring=[0.0, 0.0, 1.0], individuals=False).prepare()
        >>> system.simulate(3, random_state=0).get_values('population.count')[-1]
        40.0
    """

    field = 'mathematics/stochastic_processes'
    topics = ('branching process', 'agents', 'extinction', 'criticality')
    references = ('Watson and Galton 1875', 'Gros 2015', 'Jensen 2023')
    until = 20

    def __init__(
        self,
        n_individuals: int = 10,
        *,
        m: float = 1.0,
        offspring: str | Sequence[float] = 'poisson',
        individuals: bool = True,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_individuals = n_individuals
        self.m = m
        self.offspring = offspring
        self.individuals = individuals

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the first generation and its reproduction, by individual or by count."""
        super()._prepare(context)
        if self.individuals:
            element: Element = IndividualElement('population', size=self.n_individuals)
            reproduction: Interaction = BranchingInteraction(
                'reproduction',
                target='population',
                m=self.m,
                offspring=self.offspring,
            )
        else:
            element = GenerationElement('population', count=float(self.n_individuals))
            reproduction = GenerationInteraction(
                'reproduction',
                target='population',
                m=self.m,
                offspring=self.offspring,
            )
        self.system_ = System(self.label, elements=[element], interactions=[reproduction])
