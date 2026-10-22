"""Stochastic process models: sample-space-reducing processes and fragmentation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ....core.system import System


class ReducingBall(Element):
    """Ball that jumps to a random lower state, and restarts from the top state after it reaches state 1.

    With probability `lam` the ball jumps uniformly to a state below its own, and otherwise uniformly to any state, so
    the states are visited with probabilities proportional to `i^(-lam)`.

    Args:
        label:
            The label of the agent or group.

        level:
            The initial state, from 1 to `n_states`.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import ReducingBall
        >>> ReducingBall.group('balls', 3, n_states=10.0)
        Group('balls', element=ReducingBall, size=3)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'level'})
    level: float = state(default=10.0, domain=Domain.NONNEGATIVE_INTEGER)
    n_states: float = parameter(10.0, domain=Domain.POSITIVE)
    lam: float = parameter(1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the jump rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.jump)]

    def jump(self, agent: AgentView, model: ModelView) -> None:
        """Jump to a lower state, to any state, or back to the top state.

        Args:
            agent:
                The ball.

            model:
                The model.
        """
        top = round(agent.n_states)
        if agent.level <= 1:
            agent.level = float(top)
        elif model.rng.random() < agent.lam:
            agent.level = float(model.rng.integers(1, round(agent.level)))
        else:
            agent.level = float(model.rng.integers(1, top + 1))


class Fragment(Element):
    """Fragment of integer size that breaks at a uniformly random point into two smaller fragments.

    Args:
        label:
            The label of the agent or group.

        length:
            The initial size.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import Fragment
        >>> Fragment.group('fragments', 1, length=100.0)
        Group('fragments', element=Fragment, size=1)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'length'})
    length: float = state(default=1000.0, domain=Domain.POSITIVE)
    rate: float = parameter(0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the breaking rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.split)]

    def split(self, agent: AgentView, model: ModelView) -> None:
        """Break the fragment in two with the breaking probability.

        Args:
            agent:
                The fragment.

            model:
                The model.
        """
        length = round(agent.length)
        if length <= 1 or model.rng.random() >= agent.rate:
            return
        cut = int(model.rng.integers(1, length))
        agent.length = float(cut)
        agent.create(length=float(length - cut))


@register_model(
    field='mathematics/stochastic_processes',
    tags=('sample space reduction', 'agents', 'zipf', 'power law'),
    references=('Corominas-Murtra, Hanel, and Thurner 2015', 'Thurner et al. 2018'),
    defaults={'until': 100},
)
def sample_space_reducing(n_balls: int = 20, *, n_states: int = 10, lam: float = 1.0) -> System:
    """Return sample-space-reducing processes, whose state visits follow Zipf's law `1 / i` for `lam = 1`.

    Args:
        n_balls:
            The number of independent balls, all started at the top state.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump, the exponent of the visiting distribution.

    Returns:
        system:
            The model, with the group `balls` of state `level`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import sample_space_reducing
        >>> sample_space_reducing(4).elements[0].size
        4
    """
    return System(
        components=[
            ReducingBall.group(
                'balls',
                n_balls,
                variable=True,
                level=float(n_states),
                n_states=float(n_states),
                lam=lam,
            ),
        ],
    )


@register_model(
    field='mathematics/stochastic_processes',
    tags=('fragmentation', 'agents', 'sample space reduction'),
    references=('Thurner et al. 2018',),
    defaults={'until': 30},
)
def fragmentation(size: int = 1000, *, rate: float = 0.5) -> System:
    """Return the random fragmentation of an object of integer size, whose fragments keep breaking into smaller ones.

    The total size is conserved, and every fragment ends at size 1.

    Args:
        size:
            The size of the initial object.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Returns:
        system:
            The model, with the group `fragments` of state `length`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import fragmentation
        >>> fragmentation(50).elements[0].size
        1
    """
    return System(components=[Fragment.group('fragments', 1, variable=True, length=float(size), rate=rate)])
