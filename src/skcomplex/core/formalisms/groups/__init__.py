"""Finitely generated groups, their Cayley graphs as spaces of cellular automata, and the visual of those graphs."""

from .. import Formalism, register
from ._groups import (
    FinitelyGeneratedGroup,
    automaton_group,
    finitely_presented_group,
    matrix_group,
    parse_word,
    permutation_group,
)
from ._spaces import GroupSpace
from ._visuals import CayleyVisual

__all__: list[str] = [
    'CayleyVisual',
    'FinitelyGeneratedGroup',
    'GroupSpace',
    'automaton_group',
    'finitely_presented_group',
    'matrix_group',
    'parse_word',
    'permutation_group',
]

register(Formalism('groups', depends=(), spaces=(GroupSpace,), visuals={'cayley': CayleyVisual}))
