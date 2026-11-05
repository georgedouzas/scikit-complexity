"""Define rates, interactions that add a rate and a noise to the highest time derivative of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, vector
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec, WienerProcessSpec
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, FractionalNoiseLaw, NoiseLaw, RateLaw
from ._interaction import Interaction

CONVENTIONS = ('ito', 'stratonovich', 'anti-ito')


class RateInteraction(Interaction):
    """Rate `rate(ns)` and noise `noise(ns) dW` added to the highest time derivative of a state of a target.

    On a state of order 1 the rate is its first derivative, and on a state of order `k` it adds to the `k`-th
    derivative, so rates on one state superpose. With a divisor `d` the rate and the noise are divided by `d`, the name
    of a state or parameter of the target, such as an inertia, a friction, or a capacitance, which backends can read.

    A rate, a noise, or a divisor function takes the namespace of the interaction, with the namespaces `target` and
    `sources` of the bound elements, the time `t`, and the symbols of the parameters a subclass declares. A number
    stands for a constant, and a number on a vector state applies to every component. The noise is a Wiener process per
    component and member by default, a process of a `WienerProcessSpec` to share or correlate it across states and
    members, or a fractional Brownian motion of a Hurst index.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state the rate changes.

        state:
            The name of the state of the target that the rate changes.

        rate:
            A function of the namespace that returns the rate, or a number.

        sources:
            The paths of the elements or groups the rate and the noise read but do not change.

        noise:
            A function of the namespace that returns the amplitude of the noise, a number, or `None` for no noise.

        divisor:
            The name of the state or parameter of the target that divides the rate and the noise, a function of the
            namespace of the target that returns the divisor, such as a moment of inertia `m l^2`, or `None`.

        process:
            The Wiener processes that drive the noise, or `None` for one independent process per component and member.

        index:
            The process of `process` that drives the noise, or `None` for every process, one per component of a vector
            state.

        convention:
            The stochastic calculus of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

        hurst:
            The Hurst index in `(0, 1)` of a fractional Brownian motion that drives the noise, a number or a function
            of the namespace that returns a parameter, or `None` for a Wiener process.

        conserved:
            Whether the noise is the divergence of a random flux on the links of the lattice of a field, so it keeps the
            total of the state.

    Raises:
        ValidationError:
            The state is not a name, a function is neither callable nor a number, the convention is unknown, the Hurst
            index is outside `(0, 1)`, or the noise options cannot combine.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.interactions import RateInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> decay = RateInteraction('decay', target='stock', state='n', rate=lambda s: -0.5 * s.target.n)
        >>> system = System(elements=[CompartmentElement('stock', n=8.0)], interactions=[decay])
        >>> trajectory = Simulation(system, dt=0.01).prepare().run(until=2.0).trajectory_
        >>> round(float(trajectory.values('stock.n')[-1]), 3)
        2.943
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw, NoiseLaw, FractionalNoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state', 'divisor'))],
        state: str = 'x',
        rate: Callable[[SimpleNamespace], object] | float = 0.0,
        sources: Annotated[Sequence[str], RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = (),
        noise: Callable[[SimpleNamespace], object] | float | None = None,
        divisor: str | Callable[[SimpleNamespace], object] | None = None,
        process: Annotated[WienerProcessSpec | None, RoleSpec(noise=True)] = None,
        index: int | None = None,
        convention: str = 'ito',
        hurst: Callable[[SimpleNamespace], object] | float | None = None,
        conserved: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.rate = rate
        self.sources = sources
        self.noise = noise
        self.divisor = divisor
        self.process = process
        self.index = index
        self.convention = convention
        self.hurst = hurst
        self.conserved = conserved

    def _check_functions(self) -> None:
        """Raise an error for a state that is not a name, or a function that is neither callable nor a number."""
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = (
                f'{type(self).__name__} {self.label!r} changes the state {self.state!r}, which is not a name of a '
                'state.'
            )
            raise ValidationError(error_msg)
        for name in ('rate', 'noise'):
            value = getattr(self, name)
            if not (value is None or callable(value) or isinstance(value, int | float)):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace or a number.'
                raise ValidationError(error_msg)
        if not (
            self.divisor is None
            or callable(self.divisor)
            or (isinstance(self.divisor, str) and self.divisor.isidentifier())
        ):
            error_msg = (
                f'The divisor of {self.label!r} is {self.divisor!r}. Give the name of a state or parameter, or a '
                'function of the namespace of the target.'
            )
            raise ValidationError(error_msg)

    def _check_noise(self) -> None:
        """Raise an error for an unknown convention, a Hurst index outside `(0, 1)`, or noise options that conflict."""
        if self.convention not in CONVENTIONS:
            error_msg = f'Unknown noise convention {self.convention!r}. Use one of {", ".join(CONVENTIONS)}.'
            raise ValidationError(error_msg)
        if not (
            self.hurst is None or callable(self.hurst) or (isinstance(self.hurst, int | float) and 0 < self.hurst < 1)
        ):
            error_msg = f'The Hurst index of {self.label!r} is {self.hurst!r}. Give a number in (0, 1) or a function.'
            raise ValidationError(error_msg)
        if self.index is not None and self.process is None:
            error_msg = f'{self.label!r} picks the process {self.index!r} but binds no processes. Give `process`.'
            raise ValidationError(error_msg)
        options = [
            name
            for name, given in (('hurst', self.hurst), ('process', self.process), ('conserved', self.conserved))
            if given
        ]
        if len(options) > 1:
            error_msg = f'The noise of {self.label!r} takes {" and ".join(options)}. Give at most one of them.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for a state, functions, or noise options the interaction cannot take.

        Raises:
            ValidationError:
                The type declares states, the state is not a name, a function is neither callable nor a number, the
                convention is unknown, the Hurst index is outside `(0, 1)`, or the noise options cannot combine.
        """
        super()._check()
        self._check_functions()
        self._check_noise()

    def _evaluate(self, function: Callable[[SimpleNamespace], object] | float, namespace: SimpleNamespace) -> object:
        """Return the value of a function of the namespace, or a number as it is."""
        return function(namespace) if callable(function) else function

    def _broadcast(self, value: object, state: object) -> object:
        """Return a value with one component per component of a vector state, repeating a number."""
        if isinstance(state, FreeModuleElement) and not isinstance(value, FreeModuleElement | list | tuple):
            return vector(SR, [value] * len(state))
        if isinstance(value, list | tuple):
            return vector(SR, value)
        return value

    def _divide(self, value: object, namespace: SimpleNamespace) -> object:
        """Return a value divided by the divisor of the target, or the value without a divisor."""
        if self.divisor is None:
            return value
        divisor = (
            self.divisor(self._get_member(namespace, 'target'))
            if callable(self.divisor)
            else getattr(self._get_member(namespace, 'target'), self.divisor)
        )
        return operator.truediv(value, divisor)

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the rate before the divisor, the hook a subclass overrides to derive it from its parameters."""
        return self._evaluate(self.rate, namespace)

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the noise amplitude before the divisor, the hook a subclass overrides, or `None` for no noise."""
        return None if self.noise is None else self._evaluate(self.noise, namespace)

    def _get_noise_law(self, state: object, amplitude: object, namespace: SimpleNamespace) -> BasePart:
        """Return the noise law of an amplitude, by the process, the convention, and the Hurst index."""
        if self.hurst is not None:
            hurst = self._evaluate(self.hurst, namespace)
            return FractionalNoiseLaw(state, amplitude, hurst=hurst, convention=self.convention)
        process = None
        if self.process is not None:
            handle = self._get_member(namespace, 'process').dW
            process = handle if self.index is None else handle[self.index]
        return NoiseLaw(state, amplitude, convention=self.convention, process=process, conserved=self.conserved)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the rate law of the state and its noise law, both divided by the divisor.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the target and the sources.

        Returns:
            laws:
                The rate law, and the noise law when the interaction has noise.
        """
        state = getattr(self._get_member(namespace, 'target'), self.state)
        rate = self._divide(self._broadcast(self._get_rate(namespace), state), namespace)
        laws: list[BasePart] = [RateLaw(state, rate)]
        noise = self._get_noise(namespace)
        if noise is not None:
            laws.append(self._get_noise_law(state, self._divide(noise, namespace), namespace))
        return laws
