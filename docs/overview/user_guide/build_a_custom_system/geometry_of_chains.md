# Geometry of chains

This page measures the shape of a chain of points, such as a polymer or the backbone of a protein, frame by frame.
The analysis function `chain_shape` reads an array of frames by points by dimensions, or the positions of a group of a
[trajectory](../glossary.md#trajectory), and returns one value per frame.

## The distance between the ends

The end-to-end distance of a chain is the distance between its first and its last point. Here one frame holds a chain
of two points, three units apart along one axis and four along another.

```python
import numpy as np

from skcomplex.core.analysis import chain_shape

chain = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
assert chain_shape(chain, kind='end-to-end').values.tolist() == [5.0]
chain_shape(chain, kind='end-to-end').values
```

The distance is 5.0, the hypotenuse of a right triangle with sides 3 and 4.

## The three kinds

The argument `kind` picks the measure. The call is `chain_shape(source, kind='gyration', group=None, reference=None)`.

- `'gyration'` gives the radius of gyration per frame, the root mean square distance of the points from their centre.
- `'end-to-end'` gives the distance between the first and the last point.
- `'rmsd'` gives the root mean square deviation from `reference` after the rotation that brings the frame closest to
  it.

The frames below hold a bent chain of four points, the same chain turned a quarter turn about the vertical axis, and a
chain whose last point has moved. The deviation compares each frame with the first.

```python continuation
native = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
moved = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.5]])
frames = np.array([native, native @ turn, moved])
(
    chain_shape(frames, kind='rmsd', reference=native).values.round(3).tolist(),
    chain_shape(frames, kind='gyration').values.round(3).tolist(),
    chain_shape(frames, kind='end-to-end').values.round(3).tolist(),
)
```

The turned chain has the same shape as the first, so its deviation is 0.0, its radius of gyration stays 0.791, and
its ends stay 1.732 apart. The moved chain deviates by 0.309, its radius of gyration falls to 0.74, and its ends come
closer, 1.118 apart.

From a trajectory, `group` names the agent or particle group whose members form the chain, in member order, or the
dotted name of a lattice state whose last axis holds the coordinates of a point. Left as `None`, it takes the first
agent or particle group.
