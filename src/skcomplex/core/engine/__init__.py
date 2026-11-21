"""Define the engine that compiles a system and runs it.

It holds the compiler that turns the laws of a system into a problem, the problems it builds, the solvers that move them
in time, such as `RK4Solver`, the backends that evaluate them, the runtime that records a run, and the visuals that draw
it.
"""

from . import problems

__all__: list[str] = ['problems']
