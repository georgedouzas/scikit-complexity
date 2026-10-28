"""Pass that compiles a system of words, patches, or host graphs into a problem of the rewriting form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN

from ....core.elements import Element, GroupElement, HostGraphElement, PatchElement, WordElement
from ....core.laws import GraphProductionLaw, ProductionLaw, SubstitutionLaw
from ....core.system import System
from ....errors import IncompatibleLawsError, RewritingError, SkcomplexError, ValidationError
from ....specs import (
    BaseProblemSpec,
    CompiledElementSpec,
    EntrySpec,
    IssueSpec,
    LayoutSpec,
    NodeSpec,
    ParametersSpec,
    RewritingProblemSpec,
)
from ....specs._rewriting_problem_spec import Compiled
from ....types import FormKind, VariableKind
from ..._utils import declarations_of, symbol
from ...problems._graph_kernel import GraphKernel
from ...problems._patch_kernel import PatchKernel
from ...problems._word_kernel import WordKernel
from .._base import BaseCompiler
from ._flatten import FlattenPass
from ._layout import LayoutPass
from ._problem import ProblemPass

_LAWS_OF = {'sequence': ProductionLaw, 'tiling': SubstitutionLaw, 'graph': GraphProductionLaw}


_PRODUCTION_LAWS = (ProductionLaw, SubstitutionLaw, GraphProductionLaw)


_STRUCTURES = {WordElement: 'sequence', PatchElement: 'tiling', HostGraphElement: 'graph'}


type Production = ProductionLaw | SubstitutionLaw | GraphProductionLaw


class RewritingCompiler(BaseCompiler):
    """Compiler of a system of words, patches, or host graphs into a problem of the rewriting form.

    The backend, the solver, and the initial state do not change a problem of the rewriting form.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import RewritingCompiler
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> RewritingCompiler().check().prepare(System(elements=[PopulationElement('rabbits')])).claims_
        False
    """

    def _claims(self, system: System) -> bool:
        """Check whether a system holds a word, a patch, or a host graph.

        Args:
            system:
                The system.

        Returns:
            claims:
                Whether the compiler compiles the system.
        """
        return self._is_rewriting(system)

    def _issues(self, system: System) -> list[IssueSpec]:
        """Return the issue of a claimed system whose tree cannot be flattened.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues, none for a system the compiler does not claim.
        """
        if not self.claims_:
            return []
        try:
            FlattenPass._flatten_system(system)
        except SkcomplexError as error:
            fix = 'Pass elements, groups, interactions, and systems as children.'
            return [self._issue_of(error, system.label, 'structure', fix)]
        return []

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblemSpec:
        """Compile the prepared rewriting system into a problem of the rewriting form.

        Args:
            initial_state:
                Initial values by dotted state name, which the rewriting form does not use.

        Returns:
            problem:
                The problem of the rewriting form.
        """
        del initial_state
        return self._compile_rewriting(self.system_)

    @staticmethod
    def _is_rewriting(system: System) -> bool:
        """Check whether a system holds a word, a patch, or a host graph.

        Args:
            system:
                The system.

        Returns:
            found:
                Whether one of its elements, at any depth, is rewritten.
        """
        return any(
            RewritingCompiler._structure_of(RewritingCompiler._type_of(element)) for element in system.elements
        ) or any(RewritingCompiler._is_rewriting(child) for child in system.systems)

    @staticmethod
    def _type_of(component: object) -> type:
        """Return the element type of a group, or the type of any other component."""
        return component.element if isinstance(component, GroupElement) else type(component)

    @staticmethod
    def _productions(node: NodeSpec, namespace: SimpleNamespace) -> list[Production]:
        """Return the checked productions, substitutions, or graph productions of a word, a patch, or a host graph."""
        laws = node.component.run(namespace)
        wrong = [law for law in laws if not isinstance(law, _PRODUCTION_LAWS)]
        if wrong:
            error_msg = (
                f'{node.path} returns {type(wrong[0]).__name__} from `_get_laws`, but a word, a patch, or a host graph '
                'changes by its productions only. Return ProductionLaw, SubstitutionLaw, or GraphProductionLaw laws.'
            )
            raise RewritingError(error_msg)
        return [law.check() for law in laws]

    @staticmethod
    def _parameters(
        nodes: list[NodeSpec],
    ) -> tuple[list[EntrySpec], list[float], dict[str, SimpleNamespace], dict[str, Any]]:
        """Return the parameter entries and values of the rewritten elements, with their namespaces and symbols."""
        entries: list[EntrySpec] = []
        values: list[float] = []
        namespaces: dict[str, SimpleNamespace] = {}
        symbols: dict[str, Any] = {}
        for node in nodes:
            names = {}
            for name, declaration in declarations_of(type(node.component), VariableKind.PARAMETER).items():
                value = LayoutPass._value_of(node, name)
                if np.ndim(value) or not EntrySpec._contains(declaration.domain, value):
                    error_msg = (
                        f'{node.path}.{name} = {value!r} is not a number in the {declaration.domain.value} domain.'
                    )
                    raise ValidationError(error_msg)
                dotted = f'{node.path}.{name}'
                names[name] = symbol(f'{name}_{node.path}'.replace('/', '_').replace('-', '_'))
                symbols[dotted] = names[name]
                entries.append(EntrySpec(dotted, len(entries), (), declaration.domain))
                values.append(float(value))
            namespaces[node.path] = SimpleNamespace(**names, t=symbol('t'))
        return entries, values, namespaces, symbols

    @staticmethod
    def _compiled(
        node: NodeSpec,
        laws: list[Production],
        parameters: ParametersSpec,
        values: npt.NDArray[np.float64],
    ) -> Compiled:
        """Return the compiled rewriting system of one element from the laws that target it."""
        element = node.component
        kind = RewritingCompiler._structure_of(type(element))
        if kind is None:
            error_msg = f'{node.path} holds no sequence, tiling, or graph that rewriting laws rewrite.'
            raise ValidationError(error_msg)
        expected = _LAWS_OF[kind]
        wrong = [law for law in laws if not isinstance(law, expected)]
        if wrong:
            error_msg = (
                f'{type(element).__name__} {node.path!r} is rewritten by {expected.__name__}s, but got {wrong[0]!r}.'
            )
            raise RewritingError(error_msg)
        if kind == 'sequence':
            return WordKernel(
                node.path,
                element,
                [law for law in laws if isinstance(law, ProductionLaw)],
                parameters,
                values,
            ).check()
        if kind == 'tiling':
            return PatchKernel(
                node.path,
                element,
                [law for law in laws if isinstance(law, SubstitutionLaw)],
                parameters,
            ).check()
        return GraphKernel(node.path, element, [law for law in laws if isinstance(law, GraphProductionLaw)]).check()

    @staticmethod
    def _structure_of(element_type: type) -> str | None:
        """Return the kind of structure an element type rewrites, or `None` for a type that rewrites none."""
        if not isinstance(element_type, type):
            return None
        return next((kind for base, kind in _STRUCTURES.items() if issubclass(element_type, base)), None)

    @staticmethod
    def _compile_rewriting(system: System) -> 'RewritingProblemSpec':
        """Compile a system of words, patches, or host graphs, each rewritten by the rewriting laws it returns.

        Args:
            system:
                The system.

        Returns:
            problem:
                The problem of the rewriting form, whose states are the counts and sizes of the structures.

        Raises:
            IncompatibleLawsError:
                The system has elements or interactions that do not take part in rewriting.

            RewritingError:
                A rewritten element returns a law that is not a rewriting law, or a production, substitution, or graph
                production is malformed.

        Examples:
            >>> from skcomplex.core.spaces import DiscreteTime
            >>> from skcomplex.core.system import System
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.core.laws import ProductionLaw
            >>> from skcomplex.core.elements import WordElement
            >>> class AlgaeElement(WordElement):
            ...     def _get_laws(self, namespace):
            ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
            >>> problem = compile_system(System(elements=[AlgaeElement('algae', axiom='A')], time=DiscreteTime()))
            >>> problem.form.value, problem.states.names
            ('rewriting system', ['algae.A', 'algae.B', 'algae.size'])
        """
        nodes = FlattenPass._flatten_system(system)
        elements = {
            node.path: node
            for node in nodes
            if RewritingCompiler._structure_of(RewritingCompiler._type_of(node.component))
        }
        for node in nodes:
            if isinstance(node.component, GroupElement) and RewritingCompiler._structure_of(node.component.element):
                error_msg = f'{node.path} is a group of rewritten elements. Declare one element per structure.'
                raise RewritingError(error_msg)
            if node.path not in elements:
                error_msg = (
                    f'{node.path} takes no part in rewriting, but its system holds words, patches, or host graphs. A '
                    'rewriting system holds only rewritten elements, which rewrite themselves by their laws.'
                )
                raise IncompatibleLawsError(error_msg)
        entries, values, namespaces, symbols = RewritingCompiler._parameters(nodes)
        parameters = ParametersSpec(tuple(symbols.values()), symbol('t'))
        initial_parameters = np.asarray(values, dtype=float)
        compiled = tuple(
            RewritingCompiler._compiled(
                node,
                RewritingCompiler._productions(node, namespaces[path]),
                parameters,
                initial_parameters,
            )
            for path, node in elements.items()
        )
        states: list[EntrySpec] = []
        initial: list[float] = []
        compiled_elements = []
        for item in compiled:
            names = [*item.counted, 'size']
            component = cast('Element', elements[item.path].component)
            compiled_elements.append(
                CompiledElementSpec(
                    path=item.path,
                    element=component,
                    states={name: f'{item.path}.{name}' for name in names},
                    parameters={name: f'{item.path}.{name}' for name in vars(namespaces[item.path]) if name != 't'},
                    visual=ProblemPass._visual_of(elements[item.path]),
                ),
            )
            for name in names:
                states.append(EntrySpec(f'{item.path}.{name}', len(states), (), NN))
            initial += item.counts(item._initial(initial_parameters))  # type: ignore[arg-type]
        return RewritingProblemSpec(
            form=FormKind.REWRITING,
            states=LayoutSpec(tuple(states)),
            parameters=LayoutSpec(tuple(entries)),
            initial_state=np.asarray(initial, dtype=float),
            initial_parameters=initial_parameters,
            time=symbol('t'),
            symbols=symbols,
            elements=tuple(compiled_elements),
            system=system,
            rewriting=compiled,
        )
