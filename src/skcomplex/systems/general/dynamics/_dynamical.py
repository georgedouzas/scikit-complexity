"""Define the system of states and the equations that evolve them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.components.elements import BaseElement, DeclaredElement
from ....core.components.interactions import BaseInteraction, MapInteraction, RateInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from .types import Equations, Noises

TIMES = ('continuous', 'discrete')


class DynamicalSystem(BaseSystem[CompositeSystem]):
    """A system of states and the equations that evolve them, as rates or as maps.

    The states and parameters are the variables of one `DeclaredElement`. In continuous time the equation of a state is
    its rate, a `RateInteraction` that may carry a noise amplitude in the Itô or Stratonovich convention. In discrete
    time it is the next value of the state, a `MapInteraction`, and every map reads the states at the start of the step.
    An equation is a function of a namespace whose `target` holds the states and parameters and whose `t` holds the
    time.

    A subclass takes the parameters of its model under its own names and returns the states, equations, parameters, and
    noises from the hooks `_get_states`, `_get_equations`, `_get_parameters`, and `_get_noises`, the time from
    `_get_time`, the label of the element from `_get_element_label`, and the specs of the variables from
    `_get_variables`.

    Unlike `CompositeSystem`, which takes any parts, it is written with one `DeclaredElement` and one `RateInteraction`
    or `MapInteraction` per state. Choose it when a model is a set of equations, one per state.

    It appears in many fields. The Lorenz flow of mathematics, the populations of Wilson and Cowan in neuroscience, the
    Hénon map, and the multiplier and accelerator model of economics are dynamical systems.

    Attributes:
        system_:
            The composite of the element and its equations built from the parameters at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        The excitatory and inhibitory populations of Wilson and Cowan, which settle at a rest point:

        >>> from sage.all__sagemath_symbolics import exp
        >>> from skcomplex.systems.general.dynamics import DynamicalSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> def response(u):
        ...     return 1 / (1 + exp(-u))
        >>> populations = DynamicalSystem(
        ...     states={'e': 0.1, 'i': 0.05},
        ...     equations={
        ...         'e': lambda s: -s.target.e + response(s.target.c_ee * s.target.e - s.target.c_ei * s.target.i),
        ...         'i': lambda s: -s.target.i + response(s.target.c_ie * s.target.e - s.target.c_ii * s.target.i),
        ...     },
        ...     parameters={'c_ee': 2.0, 'c_ei': 3.0, 'c_ie': 2.0, 'c_ii': 1.0},
        ...     element='populations',
        ... )
        >>> trajectory = simulate(populations, until=50.0, dt=1.0)
        >>> e, i = (float(trajectory.get_values(f'populations.{name}')[-1]) for name in 'ei')
        >>> abs(e - float(response(2.0 * e - 3.0 * i))) < 1e-6
        True

        The multiplier-accelerator of Samuelson, whose income `y` of a period depends on the two before it:

        >>> economy = DynamicalSystem(
        ...     states={'y': 1.0, 'previous': 1.0},
        ...     equations={
        ...         'y': lambda s: s.target.g + s.target.c * (1 + s.target.v) * s.target.y - s.target.c * s.target.v
        ...         * s.target.previous,
        ...         'previous': lambda s: s.target.y,
        ...     },
        ...     parameters={'g': 1.0, 'c': 0.5, 'v': 0.5},
        ...     time='discrete',
        ...     element='economy',
        ... )
        >>> trajectory = simulate(economy, until=3)
        >>> trajectory.get_values('economy.y').tolist()
        [1.0, 1.5, 1.875, 2.03125]
        >>> sorted(economy.get_tags().laws)
        ['update']
    """

    def __init__(
        self,
        states: Mapping[str, object] | None = None,
        equations: Equations | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        noises: Noises | None = None,
        time: str = 'continuous',
        convention: str = 'ito',
        element: str = 'state',
        label: str = 'system',
    ) -> None:
        """Store the states, equations, parameters, noises, time, and labels.

        Args:
            states:
                The initial value of every state, by name, or `None` for no states.

            equations:
                The rate or the next value of every state, by name, as a function of the namespace, or `None`.

            parameters:
                The value of every parameter the equations read, by name, or `None` for none.

            noises:
                The noise amplitude of every state that has one, by name, as a function of the namespace or a
                number, or `None` for none. Only continuous time takes noise.

            time:
                The time of the equations, `'continuous'` for rates or `'discrete'` for maps.

            convention:
                The stochastic calculus of the noises, `'ito'` or `'stratonovich'`.

            element:
                The label of the element that holds the states and the parameters.

            label:
                The label of the system, which names it when it is nested in another one.
        """
        super().__init__(label)
        self.states = states
        self.equations = equations
        self.parameters = parameters
        self.noises = noises
        self.time = time
        self.convention = convention
        self.element = element

    def _get_states(self) -> Mapping[str, object]:
        """Return the initial value of every state by name.

        A subclass with parameters of its own returns it.
        """
        return self.states or {}

    def _get_equations(self) -> Equations:
        """Return the rate or the next value of every state by name.

        A subclass returns them.
        """
        return self.equations or {}

    def _get_parameters(self) -> Mapping[str, object]:
        """Return the value of every parameter by name.

        A subclass with parameters of its own returns them.
        """
        return self.parameters or {}

    def _get_noises(self) -> Noises:
        """Return the noise amplitude of every state that has one.

        A subclass with a noise returns it.
        """
        return self.noises or {}

    def _get_time(self) -> str:
        """Return the time of the equations, `'continuous'` or `'discrete'`.

        A subclass of maps returns `'discrete'`.
        """
        return self.time

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states and the parameters.

        A subclass returns the label of its model.
        """
        return self.element

    def _get_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the spec of every state and parameter, the states first.

        The states are plain `StateSpec` and the parameters plain `ParameterSpec`. A subclass whose variables have
        domains, quantities, or orders returns their specs.
        """
        states = {name: DeclaredVariableSpec(value, StateSpec()) for name, value in self._get_states().items()}
        parameters = {
            name: DeclaredVariableSpec(value, ParameterSpec()) for name, value in self._get_parameters().items()
        }
        return {**states, **parameters}

    def _check_equations(self) -> None:
        """Raise an error for an unknown time, an unknown state, or noise in discrete time.

        Raises:
            ValidationError:
                The time is unknown, an equation or a noise names a state the system does not declare, or a noise is
                given in discrete time.
        """
        time = self._get_time()
        if time not in TIMES:
            error_msg = f'Unknown time {time!r}. Use one of {", ".join(TIMES)}.'
            raise ValidationError(error_msg)
        states = [name for name, variable in self._get_variables().items() if isinstance(variable.spec, StateSpec)]
        noises = self._get_noises()
        for kind, names in (('equation', self._get_equations()), ('noise', noises)):
            unknown = [name for name in names if name not in states]
            if unknown:
                error_msg = (
                    f'{type(self).__name__} has a {kind} for the state {unknown[0]!r}, which it does not declare. '
                    f'The declared states are {states}.'
                )
                raise ValidationError(error_msg)
        if noises and time == 'discrete':
            error_msg = (
                f'{type(self).__name__} has noise amplitudes in discrete time. Noise amplitudes need continuous time.'
            )
            raise ValidationError(error_msg)

    def _get_equation_interactions(self) -> Sequence[RateInteraction | MapInteraction]:
        """Return a rate or a map for every equation, in the order of the equations.

        A rate is labelled `<state>_rate` and a map `<state>_map`.

        Returns:
            interactions:
                The equations of the states as interactions on the element.
        """
        element = self._get_element_label()
        noises = self._get_noises()
        if self._get_time() == 'discrete':
            return [
                MapInteraction(f'{state}_map', target=element, state=state, function=equation)
                for state, equation in self._get_equations().items()
            ]
        equations = self._get_equations()
        noisy = [state for state in noises if state not in equations]
        return [
            RateInteraction(
                f'{state}_rate',
                target=element,
                state=state,
                rate=equations.get(state, 0.0),
                noise=noises.get(state),
                convention=self.convention,
            )
            for state in [*equations, *noisy]
        ]

    def _get_element(self) -> BaseElement:
        """Return the element that holds the states and the parameters.

        It is a `DeclaredElement` here, and a field on the sites of a space in the subclasses that place the states.
        """
        return DeclaredElement(self._get_element_label(), variables=self._get_variables())

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the interactions of the system, the equations here.

        The subclasses that place the states add their couplings or their diffusion.
        """
        return list(self._get_equation_interactions())

    def _build_system(self) -> CompositeSystem:
        """Return the element of the states and the parameters, and their equations.

        Returns:
            system:
                The composite of the element and its equations.
        """
        self._check_equations()
        return CompositeSystem(self.label, elements=[self._get_element()], interactions=self._get_interactions())
