# Geometry of chains

Measure the shape of a chain of points, such as a polymer or a protein backbone, frame by frame. `ShapeAnalysis` reads
frames by points by dimensions, from an array or a trajectory.

```python
import numpy as np

from skcomplex.core.analysis import ShapeAnalysis

chain = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
assert ShapeAnalysis('end-to-end').prepare(chain).run().values_.tolist() == [5.0]
ShapeAnalysis('end-to-end').prepare(chain).run().values_
```

`ShapeAnalysis(kind, group=None, reference=None)` takes one of three kinds.

- `'gyration'` gives the radius of gyration per frame.
- `'end-to-end'` gives the distance between the first and the last point.
- `'rmsd'` gives the root mean square deviation from `reference` after optimal superposition.
