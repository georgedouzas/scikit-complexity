"""Test the `base` module."""

import inspect

import numpy as np
import rerun as rr

from skcomplex.core.engine import viz as package
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import BaseObserver
from skcomplex.core.engine.viz import AgentsVisual, BaseVisual, ImageVisual, PointsVisual, ScalarsVisual
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.social.opinion_dynamics import VoterSystem


def test_base_visual_get_colors_of():
    """Test `BaseVisual._get_colors_of` on a small case."""
    assert BaseVisual._get_colors_of(np.array([0.0, 1.0]), 0.0, 1.0).shape == (2, 3)


def test_base_visual_get_palette_colors():
    """Test `BaseVisual._get_palette_colors` on a small case."""
    assert BaseVisual._get_palette_colors(np.array([0.0, 1.0]), 0.0, 1.0).shape == (2, 3)


def test_base_visual_log_scalars():
    """Test `BaseVisual._log_scalars` on a small case."""
    BaseVisual._log_scalars(rr.RecordingStream('example'), 'value', 1.0 + 2.0j)


def test_public_names_are_classes():
    """Every public name is a class."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)), name


def test_classes_derive_from_a_base():
    """Every visual derives from the visual base, and the Rerun observer from the observer base."""
    for name in package.__all__:
        base = BaseObserver if name == 'RerunObserver' else package.BaseVisual
        assert issubclass(getattr(package, name), base), name


def test_verbs_take_one_input():
    """A visual prepares on a problem and runs on a frame."""
    assert list(inspect.signature(package.BaseVisual.prepare).parameters) == ['self', 'problem']
    assert list(inspect.signature(package.BaseVisual.run).parameters) == ['self', 'frame', 'final']


def test_views_of_builtin_visuals():
    """Test that the scalars, 3D points, image, and network agents visuals declare their views."""
    lotka = PassCompiler().prepare(LotkaVolterraSystem().prepare().system_).run().problem_
    gas = PassCompiler().prepare(IdealGasSystem(n_molecules=10, n_dim=3).prepare().system_).run().problem_
    gray = PassCompiler().prepare(GrayScottSystem(shape=(16, 16)).prepare().system_).run().problem_
    voters = PassCompiler().prepare(VoterSystem().prepare().system_).run().problem_
    views = [
        ScalarsVisual(path=lotka.elements[0].path).prepare(lotka).views_[0],
        PointsVisual(position='x', states=(), path=gas.elements[0].path).prepare(gas).views_[0],
        ImageVisual(path=gray.elements[0].path).prepare(gray).views_[0],
        AgentsVisual(path=voters.agents[0].path).prepare(voters).views_[0],
    ]
    assert [type(view).__name__ for view in views] == ['TimeSeriesView', 'Spatial3DView', 'Spatial2DView', 'GraphView']
    assert [str(view.origin) for view in views] == ['prey', 'gas', 'cells/u', 'voters']


def test_views_of_user_visual_are_empty_by_default():
    """Test that a visual that declares no views is left out of the layout."""
    problem = PassCompiler().prepare(LotkaVolterraSystem().prepare().system_).run().problem_
    assert BaseVisual(path=problem.elements[0].path).prepare(problem).views_ == []
