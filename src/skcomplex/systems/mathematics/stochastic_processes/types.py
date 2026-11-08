"""Define the types of the stochastic processes.

The types are the matrices of their sources.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

type Matrix = Sequence[Sequence[float]]
