"""Gather the laws that every component states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws._linear_program import _LinearProgramLaw
from .._base import BasePass
from .._compilation import Compilation
from ._census import CensusPass


class CollectPass(BasePass):
    """A pass that gathers the laws of every component.

    It takes the laws the census found and groups them by the component that states them, and keeps the linear programs
    aside, since they are solved apart from the other laws.

    Read more in the user guide, under Extend the library.

    See Also:
        CensusPass: Finds the laws.
        LinearProgramsPass: Solves the linear programs.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> compilation = PassCompiler().prepare(LotkaVolterraSystem()).compilation_
        >>> [(spec.owner, len(laws)) for spec, laws, _ in compilation.collected]
        [('prey', 0), ('predators', 0), ('growth', 1), ('predation', 2), ('death', 1)]
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('specs', 'census', 'nodes')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected', 'apart', 'programs')

    def _run(self, inputs: object = None) -> None:
        """Collect the laws of every kernel spec."""
        compilation = self.compilation_
        specs = Compilation._get_required(compilation.specs, 'specs')
        census = Compilation._get_required(compilation.census, 'census')
        flowing = CensusPass._get_flowing(census.records)
        agents = {node.path for node in Compilation._get_required(compilation.nodes, 'nodes') if node.is_agent}
        solvers = CensusPass._get_solvers(compilation.system, (compilation.solver, ''))
        declared = []
        for spec, block in specs:
            solver, scope = solvers.get(block.node.scope, (compilation.solver, ''))
            laws = CensusPass._get_declared_laws(spec, CensusPass._get_component_of(block.node), solver, scope)
            corrected = [
                CensusPass._get_corrected(law, CensusPass._get_law_owner(spec, law, agents), flowing) for law in laws
            ]
            declared.append((spec, corrected, block))
        compilation.collected = [
            (spec, [law for law in laws if not isinstance(law, _LinearProgramLaw)], block)
            for spec, laws, block in declared
        ]
        compilation.apart = []
        compilation.programs = [
            (spec, [law for law in laws if isinstance(law, _LinearProgramLaw)]) for spec, laws, _ in declared
        ]
