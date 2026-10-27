"""Test rewriting systems: words, word geometries, patches, host graphs, and their measures."""

import time

import networkx as nx
import numpy as np
import pytest
import shapely

from skcomplex.analysis import box_counting_dimension
from skcomplex.core.base import (
    BaseInteraction,
    BaseLaw,
    Form,
    IncompatibleLawsError,
    LawKind,
    ValidationError,
    parameter,
    symbol,
)
from skcomplex.core.elements import HostGraphElement, PatchElement, WordElement
from skcomplex.core.errors import RewritingError
from skcomplex.core.laws import GraphProductionLaw, ProductionLaw, SubstitutionLaw
from skcomplex.core.specs import AffineSpec, ModuleSpec, PrototileSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import PatchStructure, WordGeometry, WordStructure, claiming_compiler, compile_system
from skcomplex.engine.compile.passes import RewritingCompiler, RewritingProblem
from skcomplex.engine.problems import BaseProblem, Problem
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RewritingSolver, StructureState
from skcomplex.models.biology.ecology import PopulationElement

X = symbol('x')


class GrowthElement(WordElement):
    """Word with the parametric production `A(x) → F(x) A(x·R)` while `x > 0.1`."""

    R: float = parameter(0.5)

    def laws(self, s):
        """Return the rewriting law of the production."""
        production = ProductionLaw(
            ModuleSpec('A', (X,)), [ModuleSpec('F', (X,)), ModuleSpec('A', (X * s.R,))], condition=X > 0.1
        )
        return [RewriteLaw([production])]


class AlgaeElement(WordElement):
    """Word with the productions `A → AB` and `B → A` of the algae of Lindenmayer."""

    def laws(self, s):
        """Return the rewriting law of the productions."""
        return [RewriteLaw([ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])]


def _system(element, *rules):
    """Return a system of one element of a type that returns a rewriting law of fixed rules from its laws."""
    kind = type(
        f'Rewritten{type(element).__name__}',
        (type(element),),
        {'laws': lambda self, s: [RewriteLaw(list(rules))]},
    )
    return System(elements=[kind(**element.get_params(deep=False))])


def _fibonacci():
    """Return the L-system of the algae of Lindenmayer."""
    return _system(WordElement('word', axiom='A'), ProductionLaw('A', 'AB'), ProductionLaw('B', 'A'))


def _subdivision():
    """Return the production that replaces an edge by a path of two edges."""
    interface = nx.Graph()
    interface.add_nodes_from('ab')
    return GraphProductionLaw(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))


def test_generations_as_steps():
    """Test that an L-system compiles to the rewriting form and rewrites once per frame."""
    system = _fibonacci()
    assert compile_system(system).form == Form.REWRITING
    trajectory = Simulation(system).run(until=5).trajectory_
    assert trajectory.times.tolist() == [0, 1, 2, 3, 4, 5]
    assert trajectory.values('word.size').tolist() == [1, 2, 3, 5, 8, 13]


def test_fractional_generation():
    """Test that a fractional step is refused."""
    with pytest.raises(ValidationError, match=r'dt = 0\.5 must be a positive integer'):
        Simulation(_fibonacci(), dt=0.5).run(until=2)


def test_parikh_vector():
    """Test that the symbol counts are consecutive Fibonacci numbers whose sum is the size."""
    trajectory = Simulation(_fibonacci()).run(until=10).trajectory_
    a, b = trajectory.values('word.A'), trajectory.values('word.B')
    assert np.array_equal(a[:-1], b[1:])
    assert np.array_equal(a[2:], a[1:-1] + a[:-2])
    assert np.array_equal(trajectory.values('word.size'), a + b)


def test_stochastic_choice():
    """Test that stochastic productions are chosen by their probabilities and reproduced by the seed."""
    system = _system(
        WordElement('word', axiom='F' * 10000, alphabet='F+'),
        ProductionLaw('F', 'F+F', probability=0.3),
        ProductionLaw('F', 'F', probability=0.7),
    )
    first = Simulation(system, random_state=3).run(until=1).trajectory_
    second = Simulation(system, random_state=3).run(until=1).trajectory_
    share = first.structure('word', 1).count('+') / 10000
    assert share == pytest.approx(0.3, abs=0.015)
    assert np.array_equal(first.structure('word', 1).codes, second.structure('word', 1).codes)


def test_probabilities_that_do_not_sum_to_one():
    """Test that stochastic probabilities that sum to 0.9 are refused with the symbol and the sum."""
    system = _system(
        WordElement('word', axiom='F'),
        ProductionLaw('F', 'FF', probability=0.4),
        ProductionLaw('F', 'F', probability=0.5),
    )
    with pytest.raises(RewritingError, match=r"of 'F' in word 'word' have probabilities that sum to 0\.9"):
        compile_system(system)


def test_context_with_ignored_symbols():
    """Test a left context that skips an ignored symbol."""
    system = _system(WordElement('word', axiom='BAA+A', ignore='+'), ProductionLaw('A', 'B', left='B'))
    trajectory = Simulation(system).run(until=3).trajectory_
    assert [trajectory.structure('word', frame).text for frame in range(1, 4)] == ['BBA+A', 'BBB+A', 'BBB+B']


def test_bracketed_contexts():
    """Test that contexts follow branches: a branch sees its parent on the left, and the right skips branches."""
    system = _system(
        WordElement('word', axiom='AB[C]D[E]'),
        ProductionLaw('C', 'X', left='B'),
        ProductionLaw('B', 'Y', right='D'),
        ProductionLaw('D', 'Z', left='B'),
        ProductionLaw('E', 'W', right='F'),
    )
    trajectory = Simulation(system).run(until=1).trajectory_
    assert trajectory.structure('word', 1).text == 'AY[X]Z[E]'


def test_context_takes_precedence():
    """Test that a context-sensitive production wins over a context-free one of the same symbol."""
    system = _system(WordElement('word', axiom='BAA'), ProductionLaw('A', 'C'), ProductionLaw('A', 'B', left='B'))
    assert Simulation(system).run(until=1).trajectory_.structure('word', 1).text == 'BBC'


def test_parametric_growth():
    """Test parametric productions with a condition and successor parameters of a system parameter."""
    word = GrowthElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})
    system = System(elements=[word])
    simulation = Simulation(system).run(until=6)
    trajectory = simulation.trajectory_
    assert trajectory.structure('plant', 5).text == 'F(1) F(0.5) F(0.25) F(0.125) A(0.0625)'
    assert trajectory.structure('plant', 6).text == trajectory.structure('plant', 5).text
    assert trajectory.values('plant.R')[0] == 0.5
    assert trajectory.structure('plant', 1).modules()[1] == ModuleSpec('A', (0.5,))


def test_parameter_change_between_generations():
    """Test that a production parameter changed by a schedule applies from the next generation."""
    word = GrowthElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})
    trajectory = Simulation(System(elements=[word]), schedule={'plant.R': lambda t: 0.25}).run(until=1).trajectory_
    assert trajectory.structure('plant', 1).text == 'F(1) A(0.25)'


def test_exponential_growth_stopped():
    """Test that the size guard raises before the generation that exceeds it and keeps the earlier frames."""
    system = _system(WordElement('koch', axiom='F', max_size=10**5), ProductionLaw('F', 'F+F--F+F'))
    simulation = Simulation(system)
    with pytest.raises(RewritingError, match="Word 'koch' would reach 152916 symbols at generation 8"):
        simulation.run(until=10)
    assert len(simulation.trajectory_.times) == 8


@pytest.mark.benchmark
def test_benchmark_of_a_large_word():
    """Test that one generation of a word of a million symbols takes under 0.1 s."""
    problem = compile_system(_system(WordElement('koch', axiom='F', alphabet='F+-'), ProductionLaw('F', 'F+F--F+F')))
    compiled = problem.rewriting[0]
    word = WordStructure(compiled.names, np.zeros(10**6, dtype=np.int64), np.zeros((10**6, 0)), 0)
    start = time.perf_counter()
    rewritten = compiled.rewrite(word, problem.initial_parameters, np.random.default_rng(0))
    assert time.perf_counter() - start < 0.1
    assert len(rewritten) == 8 * 10**6


def test_malformed_systems():
    """Test the errors of unknown symbols, non-symbol predecessor parameters, mixed laws, and wrong targets."""
    with pytest.raises(RewritingError, match='not in its alphabet'):
        compile_system(_system(WordElement('word', axiom='A', alphabet='A'), ProductionLaw('A', 'AB')))
    with pytest.raises(RewritingError, match='distinct symbols'):
        compile_system(
            _system(WordElement('word', axiom=[ModuleSpec('A', (1.0,))]), ProductionLaw(ModuleSpec('A', (1.0,)), 'A')),
        )
    with pytest.raises(RewritingError, match='neither a parameter'):
        compile_system(
            _system(
                WordElement('word', axiom=[ModuleSpec('A', (1.0,))]),
                ProductionLaw(ModuleSpec('A', (X,)), [ModuleSpec('A', (X * symbol('q'),))]),
            ),
        )
    with pytest.raises(IncompatibleLawsError, match='takes no part in rewriting'):
        compile_system(System(elements=[WordElement('word', axiom='A'), PopulationElement('rabbits')]))
    with pytest.raises(IncompatibleLawsError, match='takes no part in rewriting'):
        compile_system(System(elements=[WordElement('word', axiom='A')], interactions=[BaseInteraction('rules')]))
    with pytest.raises(RewritingError, match='returns ProductionLaw from `laws`'):
        compile_system(
            System(
                elements=[
                    type('BareElement', (WordElement,), {'laws': lambda self, s: [ProductionLaw('A', 'B')]})('w'),
                ],
            ),
        )
    with pytest.raises(RewritingError, match='rewritten by ProductionLaws'):
        compile_system(_system(WordElement('word', axiom='A'), SubstitutionLaw('A', [])))


def test_koch_curve_length():
    """Test the word geometry of the Koch L-system at generation 4."""
    system = _system(
        WordElement('koch', axiom='F', geometry=WordGeometry(angle=60.0, contraction=1 / 3)),
        ProductionLaw('F', 'F+F--F+F'),
    )
    trajectory = Simulation(system).run(until=4).trajectory_
    curve = trajectory.structure('koch', 4)
    segments = curve.segments()
    assert segments.shape == (256, 2, 2)
    assert curve.length() == pytest.approx((4 / 3) ** 4, abs=1e-12)
    assert np.allclose(segments[0, 0], [0, 0], atol=1e-12)
    assert np.allclose(segments[-1, 1], [1, 0], atol=1e-12)
    assert trajectory.values('koch.length')[-1] == pytest.approx((4 / 3) ** 4, abs=1e-12)


def test_branches_return():
    """Test that a popped branch returns the cursor to where the branch began."""
    curve = (
        Simulation(System(elements=[WordElement('plant', axiom='F[+F]F', geometry=WordGeometry())])).run(0).trajectory_
    )
    segments = curve.structure('plant', 0).segments()
    assert len(segments) == 3
    assert np.allclose(segments[2, 0], segments[1, 0])
    assert np.allclose(segments[1, 1], [1, 1])
    assert curve.structure('plant', 0).depths.tolist() == [0, 1, 0]


def test_three_dimensional_word_geometry():
    """Test that pitching up and rolling move the cursor out of the plane."""
    geometry = WordGeometry(angle=90.0, dimension=3)
    curve = Simulation(System(elements=[WordElement('plant', axiom='F&F/+F', geometry=geometry)])).run(0).trajectory_
    segments = curve.structure('plant', 0).segments()
    assert segments.shape == (3, 2, 3)
    assert np.allclose(segments[1, 1], [1, 0, -1])
    assert curve.structure('plant', 0).length() == pytest.approx(3.0)


def test_module_parameters_override_the_word_geometry():
    """Test that a module's first parameter overrides the step of a move and the angle of a turn."""
    word = WordElement(
        'plant',
        axiom=[ModuleSpec('F', (2.0,)), ModuleSpec('+', (90.0,)), ModuleSpec('F', (1.0,))],
        geometry=WordGeometry(angle=10.0),
    )
    segments = Simulation(System(elements=[word])).run(0).trajectory_.structure('plant', 0).segments()
    assert np.allclose(segments[1], [[2, 0], [2, 1]])


def test_word_geometry_errors():
    """Test that unknown actions and dimensions are refused."""
    with pytest.raises(RewritingError, match='has no action'):
        WordGeometry(symbols={'G': 'jump'}).meanings()
    with pytest.raises(RewritingError, match='dimension 4'):
        WordGeometry(dimension=4).meanings()


def _cantor(generations):
    """Return the trajectory of the Cantor substitution."""
    interval = PrototileSpec('interval', [(0, 0), (1, 0)])
    rule = SubstitutionLaw(
        'interval',
        [('interval', AffineSpec(scale=1 / 3)), ('interval', AffineSpec(scale=1 / 3, translation=(2 / 3, 0)))],
    )
    return Simulation(_system(PatchElement('cantor', prototiles=[interval]), rule)).run(until=generations).trajectory_


def test_cantor_intervals():
    """Test the intervals of the Cantor substitution after 5 generations."""
    patch = _cantor(5).structure('cantor', 5)
    assert patch.parts == 32
    assert patch.length() == pytest.approx((2 / 3) ** 5, abs=1e-12)
    lefts = np.sort(patch.segments()[:, 0, 0])
    expected = sorted(
        sum(digit * 3.0 ** -(place + 1) for place, digit in enumerate(digits)) for digits in np.ndindex(*(2,) * 5)
    )
    assert np.allclose(lefts, [value * 2 for value in expected], atol=1e-12)


def test_structure_of_a_generation():
    """Test that a recorded structure is a copy that later generations do not change."""
    trajectory = _cantor(4)
    patch = trajectory.structure('cantor', 2)
    assert isinstance(patch, PatchStructure)
    assert patch.parts == 4
    assert not patch.matrices.flags.writeable
    assert patch.length() == pytest.approx(4 / 9)


def test_unknown_element():
    """Test that asking for a structure that is not a rewritten element lists the rewritten elements."""
    with pytest.raises(KeyError, match='The rewritten elements of the system are: cantor'):
        _cantor(1).structure('other', 0)


def _sierpinski(generations):
    """Return the trajectory of the Sierpiński triangle."""
    height = np.sqrt(3) / 2
    triangle = PrototileSpec('triangle', [(0, 0), (1, 0), (0.5, height)])
    corners = [(0, 0), (0.5, 0), (0.25, height / 2)]
    rule = SubstitutionLaw('triangle', [('triangle', AffineSpec(scale=0.5, translation=corner)) for corner in corners])
    return Simulation(_system(PatchElement('gasket', prototiles=[triangle]), rule)).run(until=generations).trajectory_


def test_sierpinski_triangle_area():
    """Test the area of the Sierpiński triangle after 6 generations."""
    trajectory = _sierpinski(6)
    patch = trajectory.structure('gasket', 6)
    assert patch.parts == 729
    assert patch.area() == pytest.approx(0.75**6 * np.sqrt(3) / 4, abs=1e-12)
    assert trajectory.values('gasket.triangle')[-1] == 729


def test_gasket_box_counting():
    """Test the box-counting dimension of the raster of the Sierpiński triangle at generation 8."""
    grid = _sierpinski(8).structure('gasket', 8).raster(1024)
    assert box_counting_dimension(grid, [4, 8, 16, 32, 64, 128]) == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_measures_and_raster_errors():
    """Test the bounds, the number of parts, and the refused resolutions."""
    patch = _sierpinski(1).structure('gasket', 1)
    assert patch.parts == 3
    assert patch.bounds() == pytest.approx((0, 0, 1, np.sqrt(3) / 2))
    with pytest.raises(ValidationError, match='positive integer'):
        patch.raster(0)


def test_chair_tiling_covers_its_chair():
    """Test that the chair substitution tiles the chair it starts from without overlap."""
    chair = PrototileSpec('chair', [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    pieces = [
        ('chair', AffineSpec(scale=0.5)),
        ('chair', AffineSpec(scale=0.5, translation=(0.5, 0.5))),
        ('chair', AffineSpec(scale=0.5, rotation=90, translation=(2, 0))),
        ('chair', AffineSpec(scale=0.5, rotation=-90, translation=(0, 2))),
    ]
    trajectory = (
        Simulation(_system(PatchElement('chairs', prototiles=[chair]), SubstitutionLaw('chair', pieces)))
        .run(3)
        .trajectory_
    )
    patch = trajectory.structure('chairs', 3)
    assert patch.parts == 64
    union = shapely.union_all(patch.geometry())
    assert shapely.symmetric_difference(union, shapely.Polygon(chair.vertices)).area == pytest.approx(0.0, abs=1e-9)
    assert patch.area() == pytest.approx(3.0, abs=1e-9)


def test_unsubstituted_tiles_are_kept_and_named_errors():
    """Test that tiles without substitution are kept, and unknown prototiles are refused."""
    square = PrototileSpec('square', [(0, 0), (1, 0), (1, 1), (0, 1)])
    dot = PrototileSpec('dot', [(0, 0), (0.1, 0)])
    rule = SubstitutionLaw('square', [('square', AffineSpec(scale=0.5)), ('dot', AffineSpec())])
    patch = PatchElement('patch', prototiles=[square, dot])
    trajectory = Simulation(_system(patch, rule)).run(until=2).trajectory_
    assert trajectory.values('patch.dot').tolist() == [0, 1, 2]
    with pytest.raises(RewritingError, match='not one of'):
        compile_system(_system(patch, SubstitutionLaw('circle', [])))
    with pytest.raises(RewritingError, match='two substitutions'):
        compile_system(_system(patch, rule, rule))
    with pytest.raises(RewritingError, match='no prototiles'):
        compile_system(_system(PatchElement('empty')))


def test_edge_subdivision():
    """Test that subdividing every edge of a 4-cycle three times gives a 32-cycle."""
    system = _system(HostGraphElement('graph', graph=nx.cycle_graph(4)), _subdivision())
    trajectory = Simulation(system, random_state=0).run(until=3).trajectory_
    assert trajectory.values('graph.size').tolist() == [4, 8, 16, 32]
    assert nx.is_isomorphic(trajectory.structure('graph', 3).graph, nx.cycle_graph(32))


def test_dangling_edge():
    """Test that a match whose deleted node keeps an edge outside the match is not applied, and the others are."""
    host = nx.Graph()
    host.add_node('hub', label='X')
    host.add_node('leaf', label='X')
    host.add_node('other', label='Y')
    host.add_edge('hub', 'other')
    left = nx.Graph()
    left.add_node('v', label='X')
    production = GraphProductionLaw(left, nx.Graph(), nx.Graph())
    trajectory = Simulation(_system(HostGraphElement('graph', graph=host), production)).run(until=1).trajectory_
    assert sorted(trajectory.structure('graph', 1).graph.nodes) == ['hub', 'other']
    assert trajectory.values('graph.X').tolist() == [2, 1]


def test_graph_rewriting_is_reproducible():
    """Test that a seeded run orders matches reproducibly, and that directed graphs and labels are matched."""
    host = nx.DiGraph()
    host.add_edges_from([(0, 1), (1, 2), (2, 0)], label='e')
    left = nx.DiGraph()
    left.add_edge('a', 'b', label='e')
    interface = nx.DiGraph()
    interface.add_nodes_from('ab')
    right = nx.DiGraph()
    right.add_node('c', label='new')
    right.add_edges_from([('a', 'c'), ('c', 'b')], label='f')
    system = _system(HostGraphElement('graph', graph=host), GraphProductionLaw(left, interface, right))
    first = Simulation(system, random_state=1).run(until=2).trajectory_
    second = Simulation(system, random_state=1).run(until=2).trajectory_
    assert first.values('graph.new').tolist() == [0, 3, 3]
    assert nx.utils.graphs_equal(first.structure('graph', 2).graph, second.structure('graph', 2).graph)
    with pytest.raises(RewritingError, match='not nodes of both'):
        compile_system(_system(HostGraphElement('graph', graph=host), GraphProductionLaw(left, right, right)))


def test_graph_size_guard():
    """Test that a host graph that would exceed its maximum size raises the rewriting error."""
    system = _system(HostGraphElement('graph', graph=nx.cycle_graph(4), max_size=10), _subdivision())
    with pytest.raises(RewritingError, match='would reach 16 nodes at generation 2'):
        Simulation(system).run(until=3)


def test_productions_are_written():
    """Test the written form of productions, substitutions, and graph productions."""
    production = ProductionLaw(
        ModuleSpec('A', (X,)),
        [ModuleSpec('F', (X,)), ModuleSpec('A', (X / 2,))],
        left='B',
        condition=X > 0.1,
    )
    assert production.text == 'B < A(x) → F(x) A(1/2*x) : x > 0.1'
    assert ProductionLaw('F', 'F+F', right='G', probability=0.5).text == 'F > G → F+F (0.5)'
    assert SubstitutionLaw('a', [('a', AffineSpec()), ('b', AffineSpec())]).text == 'a → a, b'
    assert WordStructure(('A',), [0], np.zeros((1, 0)), 0).parts == 1


def test_rewriting_pass_claims_rewriting_systems():
    """Test that the registered rewriting pass claims a system of words and leaves a classical one."""
    assert isinstance(claiming_compiler(System(elements=[WordElement('algae', axiom='A')])), RewritingCompiler)
    assert claiming_compiler(System(elements=[PopulationElement('rabbits', n=10)])) is None


def test_rewriting_compiler_prepares_its_claim_and_runs_the_problem():
    """Test that the rewriting compiler sets `claims_` when prepared and returns the rewriting problem when run."""
    words = System(elements=[WordElement('algae', axiom='A')])
    compiler = RewritingCompiler()
    assert not compiler.prepare(System(elements=[PopulationElement('rabbits', n=10)])).claims_
    assert compiler.prepare(words).claims_
    problem = compiler.run(words, backend='auto', solver='auto', initial_state={})
    assert problem.form == Form.REWRITING


def test_registered_formalisms_list_rewriting():
    """Test that the rewriting formalism is listed with its law kind, its form, and its solver."""
    rewriting = next(item for item in registered_formalisms() if item.name == 'rewriting')
    assert {law.law_kind for law in rewriting.laws} == {LawKind.REWRITE}
    assert rewriting.forms == (Form.REWRITING,)
    assert rewriting.defaults == (RewritingSolver,)


def test_snapshot_is_a_structure_state():
    """Test that the rewriter hands its structures and measures to the simulation as a structure state."""
    snapshot = Simulation(System(elements=[AlgaeElement('algae', axiom='A')])).run(until=2).stepper_.snapshot_
    assert isinstance(snapshot, StructureState)
    assert snapshot.structures['algae'].parts == 3


def test_rewriting_problem_is_a_base_problem_and_not_the_compilers():
    """Test that a rewriting problem derives from the problem base alone."""
    assert issubclass(RewritingProblem, BaseProblem)
    assert not issubclass(RewritingProblem, Problem)


def test_rule_of_a_word():
    """Test that a word rewrites itself by the rewriting law it returns, which is a law and not an interaction."""
    trajectory = Simulation(System(elements=[AlgaeElement('algae', axiom='A')])).run(until=5).trajectory_
    assert trajectory.structure('algae', 5).text == 'ABAABABAABAAB'
    assert issubclass(RewriteLaw, BaseLaw)
    assert not issubclass(RewriteLaw, BaseInteraction)
