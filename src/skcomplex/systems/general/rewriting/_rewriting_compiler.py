"""Pass that compiles a system of words or patches into a rewriting problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, Expression

from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import BaseSpace
from ....core.components.systems import CompositeSystem
from ....core.engine._utils import declarations_of, default_of, symbol
from ....core.engine.compile import BaseCompiler, FlattenPass, IssueSpec
from ....core.engine.problems import BaseProblemSpec, CompiledElementSpec, LayoutSpec
from ....core.enums import FormKind, VariableKind
from ....core.errors import IncompatibleLawsError, RewritingError, SkcomplexError, ValidationError
from ....core.specs import EntrySpec, VisualSpec
from ._parameters_spec import _ParametersSpec
from ._patch import PatchElement
from ._patch_kernel import PatchKernel
from ._production import ProductionLaw
from ._rewriting_problem_spec import RewritingProblemSpec
from ._rewriting_solver import RewritingSolver
from ._substitution import SubstitutionLaw
from ._word import WordElement
from ._word_kernel import WordKernel

LAWS_OF = {'sequence': ProductionLaw, 'tiling': SubstitutionLaw}
PRODUCTION_LAWS = (ProductionLaw, SubstitutionLaw)
STRUCTURES = {WordElement: 'sequence', PatchElement: 'tiling'}


type Production = ProductionLaw | SubstitutionLaw


class RewritingCompiler(BaseCompiler):
    """A compiler of a system of words or patches into a problem of the rewriting form.

    The backend, the solver, and the initial state do not change a problem of the rewriting form.

    Unlike the pass compiler of the core, which lowers laws to the equations of a form, it compiles each word or patch
    to a kernel of its own and names `RewritingSolver` as the solver of its problems. Choose it for a system of words
    or patches.

    It appears in many fields. It compiles the L-systems of botany, the fractals of mathematics, and the substitution
    sequences of computer science.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.general.rewriting._rewriting_compiler import RewritingCompiler
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> RewritingCompiler().prepare(CompositeSystem(elements=[PopulationElement('rabbits')])).report_.ok
        False
    """

    def _get_issues(self, system: CompositeSystem) -> list[IssueSpec]:
        """Return the issue of a system the compiler cannot compile or flatten.

        The tree of the system cannot be flattened, or the compiler does not compile it.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues.
        """
        if not self._is_rewriting(system):
            message = f'The system {system.label} holds no word or patch, so RewritingCompiler does not compile it.'
            fix = 'Pass the compiler of the system, or none for PassCompiler.'
            return [IssueSpec('compiler', system.label, message, fix)]
        try:
            FlattenPass._flatten_system(system)
        except SkcomplexError as error:
            fix = 'Pass elements, groups, interactions, and systems as children.'
            return [self._get_issue_of(error, system.label, 'structure', fix)]
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

    def _get_solver(self, problem: BaseProblemSpec) -> RewritingSolver:
        """Return the rewriting solver, which rewrites one generation per step.

        Args:
            problem:
                The compiled problem.

        Returns:
            solver:
                The rewriting solver.
        """
        del problem
        return RewritingSolver()

    @staticmethod
    def _is_rewriting(system: CompositeSystem) -> bool:
        """Check whether a system holds a word or a patch.

        Args:
            system:
                The system.

        Returns:
            found:
                Whether one of its elements, at any depth, is rewritten.
        """
        return any(
            RewritingCompiler._get_structure_of(RewritingCompiler._get_type_of(element)) for element in system.elements
        ) or any(RewritingCompiler._is_rewriting(child) for child in system.systems)

    @staticmethod
    def _get_type_of(component: object) -> type:
        """Return the element type of a group, or the type of any other component."""
        return component.element if isinstance(component, GroupElement) else type(component)

    @staticmethod
    def _get_productions(path: str, element: Element, namespace: SimpleNamespace) -> list[Production]:
        """Return the checked productions or substitutions of a word or a patch."""
        laws = element._get_laws(namespace)
        productions = [law for law in laws if isinstance(law, PRODUCTION_LAWS)]
        wrong = [law for law in laws if not isinstance(law, PRODUCTION_LAWS)]
        if wrong:
            error_msg = (
                f'{path} returns {type(wrong[0]).__name__} from `_get_laws`, but a word or a patch '
                'changes by its productions only. Return ProductionLaw or SubstitutionLaw laws.'
            )
            raise RewritingError(error_msg)
        return [law.prepare(path) for law in productions]

    @staticmethod
    def _get_parameters(
        elements: Mapping[str, Element],
    ) -> tuple[list[EntrySpec], list[float], dict[str, SimpleNamespace], dict[str, Expression]]:
        """Return the parameter entries and values of the rewritten elements.

        Their namespaces and symbols are also returned.
        """
        entries: list[EntrySpec] = []
        values: list[float] = []
        namespaces: dict[str, SimpleNamespace] = {}
        symbols: dict[str, Expression] = {}
        for path, element in elements.items():
            names = {}
            for name, declaration in declarations_of(type(element), VariableKind.PARAMETER).items():
                value = getattr(element, name) if hasattr(element, name) else default_of(element, name)
                number = np.asarray(value)
                if number.ndim or not EntrySpec._contains(declaration.domain, value):
                    error_msg = f'{path}.{name} = {value!r} is not a number in the {declaration.domain} domain.'
                    raise ValidationError(error_msg)
                dotted = f'{path}.{name}'
                names[name] = symbol(f'{name}_{path}'.replace('/', '_').replace('-', '_'))
                symbols[dotted] = names[name]
                entries.append(EntrySpec(dotted, len(entries), (), declaration.domain))
                values.append(float(number))
            namespaces[path] = SimpleNamespace(**names, t=symbol('t'))
        return entries, values, namespaces, symbols

    @staticmethod
    def _get_compiled(
        path: str,
        element: Element,
        laws: list[Production],
        parameters: _ParametersSpec,
        values: npt.NDArray[np.float64],
    ) -> WordKernel | PatchKernel:
        """Return the compiled rewriting system of one element.

        It is compiled from the laws that target the element.
        """
        kind = RewritingCompiler._get_structure_of(type(element))
        if kind is None:
            error_msg = f'{path} holds no sequence or tiling that rewriting laws rewrite.'
            raise ValidationError(error_msg)
        expected = LAWS_OF[kind]
        wrong = [law for law in laws if not isinstance(law, expected)]
        if wrong:
            error_msg = f'{type(element).__name__} {path!r} is rewritten by {expected.__name__}s, but got {wrong[0]!r}.'
            raise RewritingError(error_msg)
        kernel: WordKernel | PatchKernel
        if isinstance(element, WordElement):
            kernel = WordKernel(
                path,
                element,
                [law for law in laws if isinstance(law, ProductionLaw)],
                parameters,
                values,
            )
        elif isinstance(element, PatchElement):
            kernel = PatchKernel(
                path,
                element,
                [law for law in laws if isinstance(law, SubstitutionLaw)],
                parameters,
            )
        else:
            error_msg = f'{path} holds no sequence or tiling that rewriting laws rewrite.'
            raise ValidationError(error_msg)
        kernel._compile()
        return kernel

    @staticmethod
    def _get_visual_of(element: Element, space: BaseSpace | None) -> VisualSpec:
        """Return the visual of a word or a patch, the drawing its class gives."""
        prepared = 'default_visual_' in vars(element)
        default: VisualSpec = (element if prepared else element.prepare(space)).default_visual_
        return default

    @staticmethod
    def _get_structure_of(element_type: type) -> str | None:
        """Return the kind of structure an element type rewrites.

        It is `None` for a type that rewrites none.
        """
        if not isinstance(element_type, type):
            return None
        return next((kind for base, kind in STRUCTURES.items() if issubclass(element_type, base)), None)

    @staticmethod
    def _compile_rewriting(system: CompositeSystem) -> 'RewritingProblemSpec':
        """Compile a system of words or patches.

        Each is rewritten by the rewriting laws it returns.

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
                A rewritten element returns a law that is not a rewriting law, or a production or a substitution is
                malformed.

        Examples:
            >>> from skcomplex.core.components.spaces import DiscreteTime
            >>> from skcomplex.core.components.systems import CompositeSystem
            >>> from skcomplex.systems.general.rewriting._rewriting_compiler import RewritingCompiler
            >>> from skcomplex.systems.general.rewriting import ProductionLaw
            >>> from skcomplex.systems.general.rewriting import WordElement
            >>> class AlgaeElement(WordElement):
            ...     def _get_laws(self, namespace):
            ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
            >>> system = CompositeSystem(elements=[AlgaeElement('algae', axiom='A')], time=DiscreteTime())
            >>> problem = RewritingCompiler().prepare(system).run().problem_
            >>> str(problem.form), problem.states.names
            ('rewriting system', ['algae.A', 'algae.B', 'algae.size'])
        """
        nodes = FlattenPass._flatten_system(system)
        rewritten = {
            node.path: node
            for node in nodes
            if RewritingCompiler._get_structure_of(RewritingCompiler._get_type_of(node.component))
        }
        for node in nodes:
            if isinstance(node.component, GroupElement) and RewritingCompiler._get_structure_of(node.component.element):
                error_msg = f'{node.path} is a group of rewritten elements. Declare one element per structure.'
                raise RewritingError(error_msg)
            if node.path not in rewritten:
                error_msg = (
                    f'{node.path} takes no part in rewriting, but its system holds words or patches. A '
                    'rewriting system holds only rewritten elements, which rewrite themselves by their laws.'
                )
                raise IncompatibleLawsError(error_msg)
        elements = {path: node.component for path, node in rewritten.items() if isinstance(node.component, Element)}
        entries, values, namespaces, symbols = RewritingCompiler._get_parameters(elements)
        parameters = _ParametersSpec(tuple(symbols.values()), symbol('t'))
        initial_parameters = np.asarray(values, dtype=float)
        compiled = tuple(
            RewritingCompiler._get_compiled(
                path,
                element,
                RewritingCompiler._get_productions(path, element, namespaces[path]),
                parameters,
                initial_parameters,
            )
            for path, element in elements.items()
        )
        states: list[EntrySpec] = []
        initial: list[float] = []
        compiled_elements = []
        for item in compiled:
            names = [*item.counted, 'size']
            component = item.word if isinstance(item, WordKernel) else item.patch
            compiled_elements.append(
                CompiledElementSpec(
                    path=item.path,
                    element=component,
                    states={name: f'{item.path}.{name}' for name in names},
                    parameters={name: f'{item.path}.{name}' for name in vars(namespaces[item.path]) if name != 't'},
                    visual=RewritingCompiler._get_visual_of(elements[item.path], rewritten[item.path].space),
                ),
            )
            for name in names:
                states.append(EntrySpec(f'{item.path}.{name}', len(states), (), NN))
            if isinstance(item, WordKernel):
                initial += item.get_counts(item._get_initial(initial_parameters))
            else:
                initial += item.get_counts(item._get_initial(initial_parameters))
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
