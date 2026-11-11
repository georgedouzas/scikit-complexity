"""Define the interaction where members of two distributions meet and change class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, BilinearLaw, RateLaw
from ....core.components.laws.utils import collision_integral
from ....core.components.spaces import BaseSpace
from ....core.enums import ElementKind
from ....core.errors import ValidationError
from ....core.specs import RoleSpec, Tags

OUTCOMES = ('conservative', 'destructive', 'proliferative')


class EncounterInteraction(Interaction):
    """An interaction where members of two distributions meet and change class.

    A distribution is a group of `NonnegativeElement` with one member per class of a trait. A candidate at class `h`
    meets the field at class `k` at the rate `rate[h, k]`, and the `outcome` says what follows:

    - `'conservative'`: the candidate moves to class `i` with the probability `kernel[h, k, i]`.
    - `'destructive'`: the candidate is removed.
    - `'proliferative'`: the candidate creates a new candidate at class `i` with the probability `kernel[h, k, i]`, or
      at its own class without a kernel.

    The rate and the kernel are functions of the namespace, which holds `namespace.candidate`, `namespace.field`, and
    the parameters. On a `PhaseSpace` encounters pair classes at the same position. Without a field the encounter is a
    linear redistribution between classes. With a `cross_section` the rate is the binary collision integral of kinetic
    theory, evaluated by a fast spectral method, which keeps the density exactly and the momentum and energy to spectral
    accuracy.

    Unlike `BilinearLaw`, the law it lowers to, it names the outcomes of an encounter and builds the rates and kernels
    from them. Choose it when the change of a population comes from binary meetings of its members.

    It appears in many fields. Traders exchange wealth when they meet, people shift their opinion after a conversation,
    immune cells destroy tumour cells they meet, and gas molecules change velocity when they collide.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        candidate:
            The path of the group of the candidate density.

        field:
            The path of the group of the field density, or `None` for a linear redistribution.

        rate:
            The function `rate(namespace)` of the encounter rate, a number or an expression, or an array `eta[h, k]`
            of them, or `eta[h]` without a field, or `None` with a cross-section.

        kernel:
            The function `kernel(namespace)` of the array `kernel[h, k, i]` of the probabilities of the class of the
            moved or created candidate, or `kernel[h, i]` without a field, or `None` for a destructive encounter or the
            candidate's own class.

        outcome:
            `'conservative'`, `'destructive'`, or `'proliferative'`.

        state:
            The name of the state of the densities.

        cross_section:
            The function `B(speed, cosine)` of arrays of the relative speed and the cosine of the deflection angle, or
            `None` for encounters at a rate.

        factor:
            A function of the namespace that returns a factor of the collision frequency, such as a function of the
            density moment, or `None` for none.

    Raises:
        ValidationError:
            The outcome is unknown, the state is no name, a function is not callable, a conservative encounter has no
            kernel, an encounter without a field is not conservative, an encounter has neither a rate nor a
            cross-section, or a cross-section comes with a rate, a kernel, a field, or an outcome.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.general.kinetics import EncounterInteraction
        >>> kernel = np.full((3, 3, 3), 1 / 3)
        >>> mixing = EncounterInteraction(
        ...     'mixing', candidate='population', field='population', rate=lambda s: 1.0, kernel=lambda s: kernel
        ... )
        >>> mixing.outcome
        'conservative'
        >>> collide = EncounterInteraction('collide', candidate='gas', cross_section=lambda speed, cosine: 1.0)
        >>> collide.rate is None
        True
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (BilinearLaw, RateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of a collision rate, or of a linear or bilinear encounter.

        The arguments choose which.
        """
        if self._get_argument('cross_section') is not None:
            return self._get_variant_tags((RateLaw,))
        return self._get_variant_tags((BilinearLaw,))

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        field: Annotated[str | None, RoleSpec(optional=True, elements=(ElementKind.ANY,), states=('state',))] = None,
        rate: Callable[[SimpleNamespace], object] | None = None,
        kernel: Callable[[SimpleNamespace], object] | None = None,
        outcome: str = 'conservative',
        state: str = 'x',
        cross_section: Callable[..., object] | None = None,
        factor: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.candidate = candidate
        self.field = field
        self.rate = rate
        self.kernel = kernel
        self.outcome = outcome
        self.state = state
        self.cross_section = cross_section
        self.factor = factor

    def _check_collision(self) -> None:
        """Raise an error for a cross-section or factor that is no function, or misused.

        A cross-section may not come with the others.
        """
        if not callable(self.cross_section):
            error_msg = f'The cross-section of {self.label!r} is {self.cross_section!r}. Give a function of two arrays.'
            raise ValidationError(error_msg)
        if self.factor is not None and not callable(self.factor):
            error_msg = f'The factor of {self.label!r} is {self.factor!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)
        others = self.rate, self.kernel, self.field
        if any(item is not None for item in others) or self.outcome != 'conservative':
            error_msg = f'The cross-section of {self.label!r} sets the rate. Give no rate, kernel, field, or outcome.'
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown outcome, a bad function, or a missing kernel.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The outcome is unknown, the state is no name, a function is not callable, a conservative encounter has
                no kernel, an encounter without a field is not conservative, an encounter has neither a rate nor a
                cross-section, or a cross-section comes with a rate, a kernel, a field, or an outcome.
        """
        super()._prepare(context)
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} reads the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        if self.cross_section is not None:
            self._check_collision()
            return
        if self.factor is not None:
            error_msg = f'The factor of {self.label!r} scales a cross-section. Give `cross_section`.'
            raise ValidationError(error_msg)
        if self.outcome not in OUTCOMES:
            error_msg = (
                f'The outcome of encounter {self.label!r} is {self.outcome!r}. Use one of {", ".join(OUTCOMES)}.'
            )
            raise ValidationError(error_msg)
        if not callable(self.rate) or (self.kernel is not None and not callable(self.kernel)):
            error_msg = f'The rate and the kernel of encounter {self.label!r} must be functions of the namespace.'
            raise ValidationError(error_msg)
        if self.outcome == 'conservative' and self.kernel is None:
            error_msg = f'The conservative encounter {self.label!r} moves candidates by a kernel. Give `kernel`.'
            raise ValidationError(error_msg)
        if self.field is None and self.outcome != 'conservative':
            error_msg = f'The {self.outcome} encounter {self.label!r} reads a density of others. Give `field`.'
            raise ValidationError(error_msg)

    def _get_linear_laws(self, candidate: Expression, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the conserved bilinear law of a redistribution without a field.

        It uses one field class of value one.
        """
        assert self.rate is not None
        rate = np.asarray(self.rate(namespace), dtype=object)
        rates = rate if rate.ndim == 0 else rate.reshape(-1, 1)
        kernel = np.asarray(self.kernel(namespace) if self.kernel is not None else (), dtype=object)
        outcomes = np.expand_dims(kernel, axis=1)
        return [BilinearLaw(candidate, candidate, None, rates[()], outcomes, conserved=True, label=self.label)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the encounter, bilinear, or collision law of the candidate density.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.candidate` and `namespace.field`.

        Returns:
            laws:
                The law of the encounters.
        """
        candidate = getattr(self._get_member(namespace, 'candidate'), self.state)
        if self.cross_section is not None:
            factor = SR(1 if self.factor is None else self.factor(namespace))
            return [RateLaw(candidate, factor * collision_integral(candidate, self.cross_section), label=self.label)]
        if self.field is None:
            return self._get_linear_laws(candidate, namespace)
        field = getattr(self._get_member(namespace, 'field'), self.state)
        assert self.rate is not None
        rate = self.rate(namespace)
        kernel = None if self.kernel is None else self.kernel(namespace)
        if self.outcome == 'destructive':
            loss = -np.asarray(rate, dtype=object)
            return [BilinearLaw(candidate, candidate, field, loss, label=self.label)]
        if self.outcome == 'proliferative':
            return [BilinearLaw(candidate, candidate, field, rate, kernel, label=self.label)]
        return [BilinearLaw(candidate, candidate, field, rate, kernel, conserved=True, label=self.label)]
