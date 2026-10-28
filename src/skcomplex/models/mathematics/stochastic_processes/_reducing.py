"""Stochastic process models: sample-space-reducing processes and fragmentation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ..._catalog import register_model


class ReducingBallElement(Element):
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
        >>> from skcomplex.models.mathematics.stochastic_processes import ReducingBallElement
        >>> group = GroupElement('balls', element=ReducingBallElement, size=3, values={'n_states': 10.0})
        >>> group.label, group.element.__name__, group.size
        ('balls', 'ReducingBallElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'level'}),
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 10.0,
        n_states: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.n_states = n_states
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the jump rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.jump)]

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


class FragmentElement(Element):
    """Fragment of integer size that breaks at a uniformly random point into two smaller fragments.

    Args:
        label:
            The label of the agent or group.

        length:
            The initial size.

        rate:
            The probability that a fragment larger than 1 breaks in a step.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import FragmentElement
        >>> group = GroupElement('fragments', element=FragmentElement, size=1, values={'length': 100.0})
        >>> group.label, group.element.__name__, group.size
        ('fragments', 'FragmentElement', 1)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'length'}),
        *,
        length: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1000.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.length = length
        self.rate = rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the breaking rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.split)]

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
        >>> sample_space_reducing(4).elements[0].check().prepare().size_
        4
    """
    return System(
        elements=[
            GroupElement(
                'balls',
                element=ReducingBallElement,
                size=n_balls,
                values={'level': float(n_states), 'n_states': float(n_states), 'lam': lam},
                variable=True,
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
        >>> fragmentation(50).elements[0].check().prepare().size_
        1
    """
    return System(
        elements=[
            GroupElement(
                'fragments',
                element=FragmentElement,
                size=1,
                values={'length': float(size), 'rate': rate},
                variable=True,
            ),
        ],
    )
