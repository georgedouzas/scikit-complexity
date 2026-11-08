# Geometry of chains

Measure the shape of a chain of points, such as a polymer or a protein backbone, frame by frame. `chain_shape` reads
frames by points by dimensions, from an array or a trajectory.

```python
import numpy as np

from skcomplex.core.analysis import chain_shape

chain = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
assert chain_shape(chain, kind='end-to-end').values.tolist() == [5.0]
chain_shape(chain, kind='end-to-end').values
```

`chain_shape(source, kind='gyration', group=None, reference=None)` takes one of three kinds.

- `'gyration'` gives the radius of gyration per frame.
- `'end-to-end'` gives the distance between the first and the last point.
- `'rmsd'` gives the root mean square deviation from `reference` after optimal superposition.
