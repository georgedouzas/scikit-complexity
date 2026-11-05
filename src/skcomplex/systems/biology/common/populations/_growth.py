"""Define growth, interactions that change a population at a rate of a growth law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, log, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import RateInteraction
from .....core.components.laws.utils import delayed
from .....core.errors import ValidationError
from .....core.specs import ParameterSpec, RoleSpec

FORMS = ('exponential', 'logistic', 'gompertz', 'power', 'bertalanffy')
LIMITED_FORMS = ('logistic', 'gompertz')


class GrowthInteraction(RateInteraction):
    """Growth or decay of a population `n` of a target by a growth law, with a proportional harvest.

    The rate of `n` is the growth of the form minus the harvest `h n`:

    - `'exponential'`: `r n`.
    - `'logistic'`: `r n (1 - (n / K) ** p)`, the generalized logistic of Richards, with `p = 1` for Verhulst's.
    - `'gompertz'`: `r n log(K / n)`.
    - `'power'`: `r n ** p`.
    - `'bertalanffy'`: `r n ** p (1 - (n / K) ** (1 - p))`, von Bertalanffy's growth towards `K`, with `p < 1`.

    With a delay `tau` above 0, the limiting factor of the logistic and the Gompertz forms reads the population
    `n(t - tau)`, as in Hutchinson's delayed logistic equation. A negative rate `r` gives decay.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose population grows.

        r:
            The growth rate.

        K:
            The capacity of the logistic, Gompertz, and von Bertalanffy forms.

        p:
            The exponent of the logistic, power, and von Bertalanffy forms.

        h:
            The harvest rate, the fraction of the population removed per unit time.

        tau:
            The delay of the limiting factor, 0 for none.

        form:
            The growth law, `'exponential'`, `'logistic'`, `'gompertz'`, `'power'`, or `'bertalanffy'`.

        state:
            The name of the state of the target that holds the population.

    Raises:
        ValidationError:
            The form is unknown, the state is not a name, a delay is given to a form without a limiting factor, or the
            von Bertalanffy exponent is 1.

    Examples:
        >>> from skcomplex.core.components.elements import CompartmentElement
        >>> from skcomplex.systems.biology.common.populations import GrowthInteraction
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> system = System(
        ...     elements=[CompartmentElement('population', n=1.0)],
        ...     interactions=[GrowthInteraction('growth', target='population', r=1.0, K=100.0, form='logistic')],
        ... )
        >>> trajectory = Simulation(system, dt=1.0).prepare().run(until=30).trajectory_
        >>> round(float(trajectory.values('population.n')[-1]), 2)
        100.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        p: Annotated[float, ParameterSpec()] = 1.0,
        h: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        form: str = 'exponential',
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.r = r
        self.K = K
        self.p = p
        self.h = h
        self.tau = tau
        self.form = form

    def _get_limited(self, namespace: SimpleNamespace) -> Expression:
        """Return the population the limiting factor reads, delayed by `tau` when the delay is above 0."""
        population = getattr(namespace.target, self.state)
        return delayed(population, namespace.tau) if self.tau else population

    def _get_capacity(self, namespace: SimpleNamespace) -> Expression:
        """Return the capacity `K`, the hook a subclass overrides for a capacity that varies."""
        capacity: Expression = namespace.K
        return capacity

    def _get_crowding(self, namespace: SimpleNamespace) -> Expression:
        """Return the crowding `(n / K) ** p` of the logistic form, which a subclass overrides to fix the exponent."""
        crowding: Expression = (self._get_limited(namespace) / self._get_capacity(namespace)) ** namespace.p
        return crowding

    def _get_growth(self, namespace: SimpleNamespace) -> Expression:
        """Return the growth of the population by the form."""
        s = namespace
        n = getattr(s.target, self.state)
        if self.form == 'logistic':
            return s.r * n * (1 - self._get_crowding(s))
        if self.form == 'gompertz':
            return s.r * n * log(self._get_capacity(s) / self._get_limited(s))
        if self.form == 'power':
            return s.r * n**s.p
        if self.form == 'bertalanffy':
            return s.r * n**s.p * (1 - (n / self._get_capacity(s)) ** (1 - s.p))
        return s.r * n

    def _check_form(self) -> None:
        """Raise an error for an unknown form."""
        if self.form not in FORMS:
            error_msg = f'Growth {self.label!r} has the form {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)

    def _check_settings(self) -> None:
        """Raise an error for a delay of a form without a limiting factor, or a von Bertalanffy exponent of 1."""
        if self.tau and self.form not in LIMITED_FORMS:
            error_msg = (
                f'Growth {self.label!r} has the delay {self.tau!r}, but the form {self.form!r} has no limiting factor '
                f'to delay. Use one of {", ".join(LIMITED_FORMS)}, or the delay 0.'
            )
            raise ValidationError(error_msg)
        if self.form == 'bertalanffy' and self.p == 1:
            error_msg = (
                f'Growth {self.label!r} has the von Bertalanffy exponent 1. Give an exponent below 1, such as 2/3.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown form, a state that is no name, or a delay or exponent the form cannot take.

        Raises:
            ValidationError:
                The type declares states, the form is unknown, the state is not a name, a delay is given to a form
                without a limiting factor, or the von Bertalanffy exponent is 1.
        """
        super()._check()
        self._check_form()
        self._check_settings()

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the rate of the population, the growth of the form minus the harvest.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The rate of the population.
        """
        n = getattr(namespace.target, self.state)
        return self._get_growth(namespace) - namespace.h * n
