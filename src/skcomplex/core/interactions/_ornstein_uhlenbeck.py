"""Define Ornstein-Uhlenbeck interactions, which relax a state towards a mean under additive or multiplicative noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, NoiseLaw, RateLaw
from ._interaction import Interaction

NOISE_FORMS = ('additive', 'multiplicative')


class OrnsteinUhlenbeckInteraction(Interaction):
    """Ornstein-Uhlenbeck process, `dX = theta * (mean - X) * dt + sigma * dW`, on a state of a target.

    With multiplicative noise the amplitude is `sigma * X`, which gives geometric Brownian motion: with a zero mean
    and the rate `theta = -mu`, the state follows `dX = mu * X * dt + sigma * X * dW`. A vector state relaxes every
    component towards the mean and gets one Wiener process per component. A subclass that derives the rate, the mean,
    or the amplitude from other parameters returns them from `_get_coefficients`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state the process drives.

        state:
            The name of the state of the target that the process drives.

        theta:
            The rate of relaxation towards the mean. A negative rate drives the state away from it.

        mean:
            The mean the state relaxes towards.

        sigma:
            The amplitude of the noise, or of the noise per unit of the state when it is multiplicative.

        noise:
            The form of the noise, `'additive'` or `'multiplicative'`.

    Raises:
        ValidationError:
            The noise form is unknown, raised by `check`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.interactions import OrnsteinUhlenbeckInteraction
        >>> process = OrnsteinUhlenbeckInteraction('ou', target='walker', noise='multiplicative').check().prepare()
        >>> x, theta, mean, sigma = var('x theta mean sigma')
        >>> namespace = SimpleNamespace(target=SimpleNamespace(x=x), theta=theta, mean=mean, sigma=sigma)
        >>> [law.value for law in process.run(namespace).laws_]
        [(mean - x)*theta, sigma*x]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'x',
        theta: Annotated[float, ParameterSpec()] = 1.0,
        mean: Annotated[float, ParameterSpec()] = 0.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        noise: str = 'additive',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.noise = noise

    def _check(self) -> None:
        """Raise an error for declared states or for an unknown noise form.

        Raises:
            ValidationError:
                The type declares states, or the noise form is unknown.
        """
        super()._check()
        if self.noise not in NOISE_FORMS:
            error_msg = f'Unknown noise form {self.noise!r} of {self.label!r}. Use one of {", ".join(NOISE_FORMS)}.'
            raise ValidationError(error_msg)

    def _get_coefficients(self, namespace: SimpleNamespace) -> tuple[Expression, Expression, Expression]:
        """Return the rate, the mean, and the noise amplitude of the process.

        Args:
            namespace:
                The namespace of the symbols the coefficients can use.

        Returns:
            coefficients:
                The rate, the mean, and the noise amplitude as symbolic expressions.
        """
        return namespace.theta, namespace.mean, namespace.sigma

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of the state towards the mean and its noise.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law and the noise law of the state.
        """
        state = getattr(namespace.target, self.state)
        rate, mean, amplitude = self._get_coefficients(namespace)
        if isinstance(state, FreeModuleElement):
            drift = state.apply_map(lambda component: rate * (mean - component))
        else:
            drift = rate * (mean - state)
        if self.noise == 'multiplicative':
            amplitude = amplitude * state
        return [RateLaw(state, drift), NoiseLaw(state, amplitude)]
