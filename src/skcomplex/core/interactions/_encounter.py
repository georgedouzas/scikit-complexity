"""Define encounters, bilinear moves of a density over activity classes by meetings with another density."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, DestructionLaw, EncounterLaw, ProliferationLaw
from ._interaction import Interaction

OUTCOMES = ('conservative', 'destructive', 'proliferative')


class EncounterInteraction(Interaction):
    """Encounters of a candidate density with a field density over activity classes, which move, remove, or create.

    The candidate and the field are groups of `DensityElement` on the sites of lattices of activity classes, one member
    per class. A candidate at class `h` meets a field at class `k` at the rate `eta[h, k]`. In the outcome
    `'conservative'` the candidate moves to class `i` with the probability `kernel[h, k, i]`, in the outcome
    `'destructive'` it is removed, and in the outcome `'proliferative'` it creates a new candidate at class `i` with
    the probability `kernel[h, k, i]`, or at its own class without a kernel. The rate and the kernel are functions of
    the namespace, which holds `namespace.candidate`, `namespace.field`, and the parameters of the interaction.

    On a `PhaseSpace` the classes are the points of the class axes, and encounters pair classes at the same position. A
    rate that is one expression may read the `coordinate` of the class of the candidate. Without a field the encounter
    is a linear redistribution: a conservative candidate at class `h` moves to class `i` at the rate `eta[h]` with the
    probability `kernel[h, i]`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        candidate:
            The path of the group of the candidate density.

        field:
            The path of the group of the field density, or `None` for a linear redistribution.

        rate:
            The function `rate(namespace)` of the encounter rate, a number or an expression, or an array `eta[h, k]`
            of them, or `eta[h]` without a field.

        kernel:
            The function `kernel(namespace)` of the array `kernel[h, k, i]` of the probabilities of the class of the
            moved or created candidate, or `kernel[h, i]` without a field, or `None` for a destructive encounter or the
            candidate's own class.

        outcome:
            `'conservative'`, `'destructive'`, or `'proliferative'`.

        state:
            The name of the state of the densities.

    Raises:
        ValidationError:
            The outcome is unknown, a function is not callable, a conservative encounter has no kernel, or an encounter
            without a field is not conservative.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.interactions import EncounterInteraction
        >>> kernel = np.full((3, 3, 3), 1 / 3)
        >>> mixing = EncounterInteraction(
        ...     'mixing', candidate='population', field='population', rate=lambda s: 1.0, kernel=lambda s: kernel
        ... )
        >>> mixing.check().outcome
        'conservative'
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str | None, RoleSpec(optional=True)] = None,
        rate: Callable[[SimpleNamespace], object],
        kernel: Callable[[SimpleNamespace], object] | None = None,
        outcome: str = 'conservative',
        state: str = 'f',
    ) -> None:
        super().__init__(label)
        self.candidate = candidate
        self.field = field
        self.rate = rate
        self.kernel = kernel
        self.outcome = outcome
        self.state = state

    def _check(self) -> None:
        """Raise an error for an unknown outcome, a function that is not callable, or a conservative kernel missing.

        Raises:
            ValidationError:
                The outcome is unknown, a function is not callable, a conservative encounter has no kernel, or an
                encounter without a field is not conservative.
        """
        super()._check()
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

    def _linear_laws(self, candidate: Expression, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the encounter law of a redistribution without a field, one field class of value one."""
        rate = np.asarray(self.rate(namespace), dtype=object)
        rates = rate if rate.ndim == 0 else rate.reshape(-1, 1)
        kernel = np.asarray(self.kernel(namespace) if self.kernel is not None else (), dtype=object)
        return [EncounterLaw(candidate, None, rates[()], np.expand_dims(kernel, axis=1), label=self.label)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the encounter, destruction, or proliferation law of the candidate density.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.candidate` and `namespace.field`.

        Returns:
            laws:
                The law of the encounters.
        """
        candidate = getattr(self._get_member(namespace, 'candidate'), self.state)
        if self.field is None:
            return self._linear_laws(candidate, namespace)
        field = getattr(self._get_member(namespace, 'field'), self.state)
        rate = self.rate(namespace)
        kernel = None if self.kernel is None else self.kernel(namespace)
        if self.outcome == 'destructive':
            return [DestructionLaw(candidate, field, rate, label=self.label)]
        if self.outcome == 'proliferative':
            return [ProliferationLaw(candidate, field, rate, kernel, label=self.label)]
        return [EncounterLaw(candidate, field, rate, kernel, label=self.label)]
