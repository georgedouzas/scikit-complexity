"""Name the array types that compiled problems hold: vectors of values and arrays of flat indices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]
