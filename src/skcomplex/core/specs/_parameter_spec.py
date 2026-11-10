"""Define the parameter spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import RR

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ParameterSpec(BaseSpec):
    """A spec of a parameter of an element or an interaction.

    It is given in the annotation of an argument of the constructor, and the default of the argument is the default of
    the parameter. A parameter per entry of a variadic role holds one value per entry of the role. An adaptive parameter
    is a state that laws may change, such as a policy rate set by a central bank.

    It appears in every model with parameters, such as the rate of infection of an epidemic, the rate constant of a
    reaction, and the coupling strength of oscillators, which calibration and sweeps change.

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

        shape:
            The names of the roles whose group sizes give the shape of the parameter, such as `('target', 'source')`
            for a matrix with one row per member of the target and one column per member of the source, or `None` for
            a scalar. A role repeated k times gives a tensor of order k.

    Examples:
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> from skcomplex.core.specs import ParameterSpec
        >>> 0.5 in ParameterSpec(domain=RealSet.open(0, oo)).domain
        True
    """

    domain: object = RR
    vector: bool = False
    per: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None
    shape: tuple[str, ...] | None = None

    def _check(self) -> None:
        """Raise an error for an invalid parameter per role, bad bounds, or a bad shape.

        A parameter per role must not be a vector or adaptive.

        Raises:
            ValidationError:
                The parameter is per entry of a role and a vector or adaptive, has bounds without being adaptive or
                with a lower bound above the upper one, or has a shape that is empty, not role names, or combined with
                `vector` or `per`.
        """
        if self.per is not None and (self.vector or self.adaptive):
            error_msg = f'A parameter per entry of the role {self.per!r} is neither a vector nor adaptive.'
            raise ValidationError(error_msg)
        if self.bounds is not None and not (self.adaptive and float(self.bounds[0]) <= float(self.bounds[1])):
            error_msg = (
                f'The bounds {self.bounds!r} need an adaptive parameter and a lower bound at most the upper one.'
            )
            raise ValidationError(error_msg)
        if self.shape is not None and (
            not self.shape or not all(isinstance(role, str) for role in self.shape) or self.vector or self.per
        ):
            error_msg = (
                f'The shape {self.shape!r} of a parameter is a nonempty tuple of role names, and a shaped parameter is '
                'neither a vector nor per entry of a role.'
            )
            raise ValidationError(error_msg)
