"""Define the types of neuroscience: the rates of the gates of conductances."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import Expression

type GateRates = dict[str, tuple[Expression, Expression]]
