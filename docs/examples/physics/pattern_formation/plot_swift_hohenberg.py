"""
# Swift-Hohenberg

Just above the onset of convection, small random noise organizes itself into parallel rolls of one preferred width,
the wavelength `2π`. Patches of rolls with different orientations meet at defects and slowly straighten out.
"""

# %%
# The field starts from noise of amplitude 0.01 on a periodic square of side `16π`, eight roll wavelengths, at the
# distance `r = 0.3` above onset. The catalog defaults integrate it with an exponential time differencing solver, and
# the default blueprint shows the field as an image.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of, pattern_wavelength, structure_factor
from skcomplex.core.viz import RerunObserver
from skcomplex.models import run

MODEL = 'swift_hohenberg'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, random_state=0, settings={'observers': [observer], 'random_state': 0})

# %%
# Selected wavelength
# -------------------
# The linear growth rate `r - (1 - k^2)^2` is largest at the wavenumber `k = 1`. The structure factor of the final
# field should peak there, and the dominant wavelength measured from the pattern should be close to `2π`.

field = fields_of(simulation.trajectory_, 'cells.u')[-1]
spacing = 16 * np.pi / field.shape[0]
factor = structure_factor(field, spacing=spacing)
pattern = pattern_wavelength(field, spacing=spacing)
fig, (image, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
image.imshow(field, cmap='viridis', extent=(0, 16 * np.pi, 0, 16 * np.pi))
image.set_xlabel('x')
image.set_ylabel('y')
image.set_title(f'Rolls of wavelength {pattern.wavelength:.2f} (2π = {2 * np.pi:.2f})')
spectrum.plot(factor.wavenumbers, factor.power, 'o-', markersize=3)
spectrum.axvline(1.0, linestyle='--', color='black', label='k = 1')
spectrum.set_xlabel('wavenumber k')
spectrum.set_ylabel('structure factor S(k)')
spectrum.set_title('The structure factor peaks at k = 1')
spectrum.legend()
fig.tight_layout()
plt.show()
