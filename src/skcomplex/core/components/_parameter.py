"""Define the record of a declared parameter."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import RR


@dataclass(frozen=True)
class _Parameter:
    """A parameter of an element or an interaction.

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

        n_dim:
            The number of components of a vector parameter, or of rows of a tensor parameter, such as `2` for the value
            `[1.0, 2.0]`, or `None` for the number of dimensions of the enclosing space.

        tensor:
            The symmetry of a rank-2 tensor parameter, `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, or
            `None` for a parameter that is no tensor. Laws read it as a Sage matrix.
    """

    domain: object = RR
    vector: bool = False
    per: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None
    shape: tuple[str, ...] | None = None
    n_dim: int | None = None
    tensor: str | None = None
