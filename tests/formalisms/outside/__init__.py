"""Formalism outside the core with its own law kind, form, compilation pass, and solver, for registration tests."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np

from skcomplex.core.base import (
    BaseLaw,
    ComponentKind,
    DomainKind,
    Entry,
    Form,
    LawClass,
    LawKind,
    Layout,
    NumericDomain,
    OrderKind,
    Requires,
    Supports,
    TimeKind,
    Trait,
    Vector,
    VisualSpec,
    parameter,
    state,
    symbol,
)
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import Symbols
from skcomplex.core.problems import CompiledElement, Problem
from skcomplex.core.system import System
from skcomplex.engine.compile import BaseSystemCompiler
from skcomplex.engine.solvers import Solver
from skcomplex.formalisms import Formalism, register

DECAY = LawKind.register('DECAY', 'decay')
DECAYING = Form.register('DECAYING', 'exponential decay')


@dataclass(frozen=True)
class DecayLaw(BaseLaw):
    """Decay of a state at a rate given by a parameter symbol.

    Args:
        target:
            The state symbol.

        rate:
            The parameter symbol of the rate.
    """

    law_kind: ClassVar[LawKind] = DECAY
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires()

    target: object
    rate: object


class NucleusComponent(ElementComponent):
    """Amount of a substance that decays.

    Args:
        label:
            The label of the nucleus.

        n:
            The initial amount.

        k:
            The decay rate.
    """

    n: float = state(default=1.0, domain=NumericDomain.NONNEGATIVE)
    k: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the decay of `n` at the rate `k`."""
        return [DecayLaw(s.n, s.k)]


@dataclass(frozen=True)
class DecayProblem(Problem):
    """Problem of the decaying form, with the parameter index of the rate of every state.

    Args:
        rates:
            The index of the rate parameter of every state.
    """

    rates: tuple[int, ...] = ()


class DecayCompiler(BaseSystemCompiler):
    """Pass that compiles a system of nuclei into a problem of the decaying form."""

    def claims(self, system: System) -> bool:
        """Check whether every element of the system is a nucleus."""
        return bool(system.elements) and all(isinstance(element, NucleusComponent) for element in system.elements)

    def compile(
        self,
        system: System,
        *,
        backend: object,
        solver: object,
        initial_state: Mapping[str, object],
    ) -> Problem:
        """Compile the nuclei into one state and one rate parameter each."""
        del backend, solver
        nuclei = list(system.elements)
        states = [Entry(f'{item.label}.n', index, (), NumericDomain.NONNEGATIVE) for index, item in enumerate(nuclei)]
        parameters = [Entry(f'{item.label}.k', index, (), NumericDomain.POSITIVE) for index, item in enumerate(nuclei)]
        symbols = {entry.name: symbol(entry.name.replace('.', '_')) for entry in [*states, *parameters]}
        columns = {symbols[entry.name]: entry.start for entry in parameters}
        laws = [item.laws(Symbols(n=symbols[f'{item.label}.n'], k=symbols[f'{item.label}.k'])) for item in nuclei]
        initial = [float(cast('float', initial_state.get(f'{item.label}.n', item.n))) for item in nuclei]
        return DecayProblem(
            form=DECAYING,
            states=Layout(states),
            parameters=Layout(parameters),
            initial_state=np.array(initial),
            initial_parameters=np.array([float(item.k) for item in nuclei]),
            time=symbol('t'),
            symbols=symbols,
            equations={},
            equation_symbols={},
            has_callables=False,
            has_group_sums=False,
            elements=tuple(
                CompiledElement(item.label, item, {'n': f'{item.label}.n'}, {'k': f'{item.label}.k'}, VisualSpec())
                for item in nuclei
            ),
            system=system,
            rates=tuple(columns[law.rate] for (law,) in laws),
        )


class ExactDecaySolver(Solver):
    """Solver that advances every state by its exact exponential decay."""

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        laws={DECAY},
        domains={DomainKind.REAL},
        components={ComponentKind.ELEMENT},
        orders={OrderKind.INTEGER},
    )
    form: ClassVar[Form] = DECAYING

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Return the states decayed from time `t` to time `until`."""
        rates = p[np.asarray(cast('DecayProblem', self.problem_).rates, dtype=np.intp)]
        return x * np.exp(-rates * (until - t))


FORMALISM = Formalism(
    'outside',
    laws=(DecayLaw,),
    components=(NucleusComponent,),
    compilers=(DecayCompiler(),),
    forms=(DECAYING,),
    solvers={'exact-decay': ExactDecaySolver},
    defaults=(ExactDecaySolver,),
)

register(FORMALISM)
