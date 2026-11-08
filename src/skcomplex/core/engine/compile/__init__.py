"""Compile systems into problems of one mathematical form with fast numeric functions.

The compilers of the library's forms are private: `simulate` chooses the one of the form a system holds. A system
package that brings a form of its own defines its compiler on `BaseCompiler`, with passes on `BasePass` and lowerings on
`BaseLowering`, and tunes it as the `compiler` run setting of its system.
"""

from ._base import BaseCompiler, BaseLowering, BasePass
from .passes._hilbert_space_error import HilbertSpaceError

__all__: list[str] = [
    'BaseCompiler',
    'BaseLowering',
    'BasePass',
    'HilbertSpaceError',
]
