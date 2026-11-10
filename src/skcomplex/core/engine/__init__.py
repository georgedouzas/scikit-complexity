"""Define the engine that compiles a system and runs it.

It holds the compiler, the problems it builds, the solvers that advance them, the backends that evaluate them, the
runtime that records a run, and the visuals that draw it.
"""

from . import problems

__all__: list[str] = ['problems']
