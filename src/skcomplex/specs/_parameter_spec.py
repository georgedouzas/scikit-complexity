"""Define the parameter spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Self

from sage.all__sagemath_symbolics import RR

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ParameterSpec(BaseSpec):
    """Spec of a parameter of an element or an interaction, in the annotation of an argument of its constructor.

    The default of the argument is the default of the parameter. A parameter per entry of a variadic role holds one
    value per entry of the role, and an adaptive parameter is a state that laws may change, such as a policy rate.

    Args:
        domain:
            The Sage set of the values of the parameter, such as `RR` or `RealSet.open(0, oo)`.

        vector:
            Whether the parameter has one component per dimension of the enclosing space.

        per:
            The name of a variadic role whose entries the parameter holds one value for, or `None`.

        adaptive:
            Whether laws may change the parameter.

        bounds:
            The lower and upper bound of an adaptive parameter that is a control input, or `None` for none.

    Examples:
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> from skcomplex.specs import ParameterSpec
        >>> 0.5 in ParameterSpec(domain=RealSet.open(0, oo)).check().domain
        True
    """

    domain: object = RR
    vector: bool = False
    per: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None

    def _check(self) -> Self:
        """Raise an error for a parameter per role that is a vector or adaptive, or for invalid bounds.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                The parameter is per entry of a role and a vector or adaptive, or has bounds without being adaptive or
                with a lower bound above the upper one.
        """
        if self.per is not None and (self.vector or self.adaptive):
            error_msg = f'A parameter per entry of the role {self.per!r} is neither a vector nor adaptive.'
            raise ValidationError(error_msg)
        if self.bounds is not None and not (self.adaptive and float(self.bounds[0]) <= float(self.bounds[1])):
            error_msg = (
                f'The bounds {self.bounds!r} need an adaptive parameter and a lower bound at most the upper one.'
            )
            raise ValidationError(error_msg)
        return self
