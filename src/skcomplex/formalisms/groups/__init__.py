"""Finitely generated groups, their Cayley graphs as spaces of cellular automata, and the visual of those graphs."""

from .. import Formalism, register
from ._groups import (
    FinitelyGeneratedGroup,
    WordProblemError,
    automaton_group,
    finitely_presented_group,
    matrix_group,
    parse_word,
    permutation_group,
)
from ._spaces import CayleySpace
from ._visuals import CayleyVisual

__all__: list[str] = [
    'CayleySpace',
    'CayleyVisual',
    'FinitelyGeneratedGroup',
    'WordProblemError',
    'automaton_group',
    'finitely_presented_group',
    'matrix_group',
    'parse_word',
    'permutation_group',
]

register(Formalism('groups', depends=(), spaces=(CayleySpace,), visuals={'cayley': CayleyVisual}))
