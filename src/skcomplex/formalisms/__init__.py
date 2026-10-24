"""Formalisms that add laws, components, compilation, solvers, backends, visuals, and checks to core and engine."""

from ._formalism import Formalism, register, registered_formalisms

__all__: list[str] = [
    'Formalism',
    'register',
    'registered_formalisms',
]
