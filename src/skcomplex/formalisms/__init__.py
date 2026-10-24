"""Formalisms that add their laws, components, compilation, solvers, backends, visuals, and checks to the core."""

from ._formalism import Formalism, register, registered_formalisms

__all__: list[str] = [
    'Formalism',
    'register',
    'registered_formalisms',
]
