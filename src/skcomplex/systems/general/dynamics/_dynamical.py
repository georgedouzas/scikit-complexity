"""Define the system of states and the equations that evolve them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Mapping, Sequence

from ....core.components import Observable
from ....core.components.elements import BaseElement, Element
from ....core.components.interactions import BaseInteraction
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ._equation import EquationInteraction
from .types import Equations, Noises

TIMES = ('continuous', 'discrete')
PRIME = "'"
STARTING = re.compile(r"(?P<name>\w+)('+|\[-\d+\])")


class DynamicalSystem(BaseSystem):
    """A system of states and the equations that evolve them, as rates or as maps.

    The states and parameters are the values of one element. In continuous time the equation of a state is its rate,
    or the rate of its derivative when the state is named with primes, such as `"x''"` for an acceleration, and it may
    carry a noise amplitude in the Itô or Stratonovich convention. In discrete time it is the next value of the state,
    and every map reads the states at the start of the step. An equation is a function of a namespace whose `target`
    holds the states and parameters as functions of the time `t`, which it also holds. A derivative or a past value
    starts from the value named with primes or a past index among the states, such as `{'y': 1.0, 'y[-1]': 1.0}`.

    A subclass takes the parameters of its model under its own names and returns the states, equations, parameters, and
    noises from `_get_states`, `_get_equations`, `_get_parameters`, and `_get_noises`, the time from `_get_time`, and
    the label of the element from `_get_element_label`.

    Unlike `System`, which takes any components, it is written with one element and one `EquationInteraction` per
    state. Choose it when a model is a set of equations, one per state.

    It appears in many fields. The Lorenz flow of mathematics, the populations of Wilson and Cowan in neuroscience, the
    Hénon map, and the multiplier and accelerator model of economics are dynamical systems.

    Attributes:
        system_:
            The `System` of the element and its equations built from the parameters at the last `prepare`.

    See Also:
        StockFlowSystem: Stocks that change only through named flows between them.

    Examples:
        The excitatory and inhibitory populations of Wilson and Cowan, which settle at a rest point:

        >>> from sage.all__sagemath_symbolics import exp
        >>> from skcomplex.systems.general.dynamics import DynamicalSystem
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
        >>> trajectory = populations.prepare().simulate(until=50.0, dt=1.0)
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
        >>> economy.prepare().simulate(until=3).get_values('economy.y').tolist()
        [1.0, 1.5, 1.875, 2.03125]
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        """Store the states, equations, parameters, noises, time, labels, and settings.

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

            solver:
                The solver of the problem, or `'auto'` for the one the system chooses.

            backend:
                The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.
        """
        super().__init__(label, solver=solver, backend=backend)
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

    def _get_solver(self) -> BaseSolver | str:
        """Return the tuned solver, or `'auto'` for the one the problem chooses.

        A subclass whose equations need a solver of their own returns it. A `solver` given to the system replaces it.
        """
        return 'auto'

    def _get_state_names(self) -> list[str]:
        """Return the names of the states, without starting derivatives and past values."""
        return [name for name in self._get_states() if STARTING.fullmatch(name) is None]

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
        states = self._get_state_names()
        noises = self._get_noises()
        for kind, names in (('equation', self._get_equations()), ('noise', noises)):
            unknown = [name for name in names if name.rstrip(PRIME) not in states]
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

    def _get_equation_interactions(self) -> Sequence[BaseInteraction]:
        """Return an equation for every state that has one, in the order of the equations.

        A rate is labelled `<state>_rate` and a map `<state>_map`, after the state without its primes.

        Returns:
            interactions:
                The equations of the states as interactions on the element.
        """
        element = self._get_element_label()
        time = self._get_time()
        equations = self._get_equations()
        noises = self._get_noises()
        suffix = 'map' if time == 'discrete' else 'rate'
        return [
            EquationInteraction(
                f'{state.rstrip(PRIME)}_{suffix}',
                target=element,
                state=state,
                equation=equations.get(state),
                noise=noises.get(state),
                convention=self.convention,
                time=time,
            )
            for state in [*equations, *(state for state in noises if state not in equations)]
        ]

    def _get_element(self) -> BaseElement:
        """Return the element that holds the states and the parameters.

        It is one element here, and a field on the sites of a space or a node of a graph in the subclasses that place
        the states.
        """
        return Element(self._get_element_label(), values={**self._get_states(), **self._get_parameters()})

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the interactions of the system, the equations here.

        The subclasses that place the states add their couplings or their diffusion.
        """
        return list(self._get_equation_interactions())

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Check the equations and set `system_` to the element and its equations.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self._check_equations()
        self.system_ = System(
            self.label,
            elements=[self._get_element()],
            interactions=self._get_interactions(),
            solver=self._get_solver(),
        )
