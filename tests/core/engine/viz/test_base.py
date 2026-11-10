"""Test the `base` module."""

import numpy as np
import rerun as rr

from skcomplex.core.engine.viz import BaseVisual


def test_base_visual_get_colors_of():
    """Test `BaseVisual._get_colors_of` on a small case."""
    assert BaseVisual._get_colors_of(np.array([0.0, 1.0]), 0.0, 1.0).shape == (2, 3)


def test_base_visual_get_palette_colors():
    """Test `BaseVisual._get_palette_colors` on a small case."""
    assert BaseVisual._get_palette_colors(np.array([0.0, 1.0]), 0.0, 1.0).shape == (2, 3)


def test_base_visual_log_scalars():
    """Test `BaseVisual._log_scalars` on a small case."""
    BaseVisual._log_scalars(rr.RecordingStream('example'), 'value', 1.0 + 2.0j)
