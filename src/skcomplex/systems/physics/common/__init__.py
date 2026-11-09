"""Realizations shared by several physics packages, by family.

The families are fields, mechanics, oscillators, particles, quantum species, and spins. The sandpile cell, which the
criticality systems share, has no family.
"""

from ._sandpile_cell import SandpileCellElement

__all__: list[str] = [
    "SandpileCellElement",
]
