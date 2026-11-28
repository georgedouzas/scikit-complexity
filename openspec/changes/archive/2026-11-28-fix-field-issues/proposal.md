# Proposal

## Why

Four issues, #45 to #48, came from the pattern formation chapter of a tutorial. The Lyapunov spectrum and the
dispersion relation fail on the Kuramoto Sivashinsky system, the total defect charge on a periodic lattice is not 0
when an angle difference is exactly half a period, and stiff reaction diffusion systems such as Schnakenberg and
Brusselator run far slower than a plain explicit stepper.

## What Changes

- **Lyapunov spectra of fields (#45):** `lyapunov(kind='spectrum')` runs on spectral and lattice fields, including
  `KuramotoSivashinskySystem` at its default 128 modes. It reads the initial state as one vector instead of by component
  name, and above the member limit of symbolic expansion it takes the Jacobian from the backend instead of from Sage.
- **Dispersion of spectral derivatives (#46):** `stability(kind='dispersion')` linearises `partial` derivatives of any
  order as well as `laplacian`, so the Kuramoto Sivashinsky growth rate is `q^2 - q^4`.
- **Defect charges (#47):** `defects` gives every edge one wrapped difference, antisymmetric between its two
  orientations, so the charges on a periodic lattice always sum to 0. The docstring says so.
- **Stiff reaction diffusion (#48):** Schnakenberg and Brusselator runs of 40 by 40 sites take seconds, not minutes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `trajectory-analysis`: Lyapunov spectra of fields, dispersion of spectral derivatives, and defect charges.
- `simulation`: fast runs of stiff reaction diffusion systems.

## Impact

- Code: `core/analysis/_lyapunov.py` or wherever `_compute_spectrum` lives, `core/analysis/_stability.py`,
  `core/analysis/_defects.py`, and the solver choice or backend kernels the profile of #48 points to.
- Behaviour: defect charges at exact half period ties change, and reaction diffusion runs may choose another solver
  by default, which changes their frames within the solver tolerance.
