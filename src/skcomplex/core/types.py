"""Define the type aliases shared across the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from .specs import ModuleSpec

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]
type Target = Expression | FreeModuleElement
type Process = str | tuple[str, ...] | Expression | FreeModuleElement | None
type Letters = str | ModuleSpec | Sequence[str | ModuleSpec]
