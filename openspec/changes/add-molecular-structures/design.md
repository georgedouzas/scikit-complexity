# Design

## Context

The parts of the current code this change builds on:
- `LatticeElement(label, element=, space=LatticeSpace(shape))` in `skcomplex.core.elements` holds one value per site.
  A state with the domain `FiniteEnumeratedSet(['A', 'C', 'G', 'U'])` is categorical, and `EntrySpec` names its labels.
- `SamplingInteraction(target, state, objective, move, intensity, rule, schedule, neighborhood, values, width)` in
  `skcomplex.core.interactions` lowers to a `SampleLaw` with an `ObjectiveSpec`, a `MoveSpec`, and a `SamplerSpec`, and
  runs on `MonteCarloSolver` under `DiscreteTime`. Its objective is a local function of the namespace, its move may be a
  function `move(values, random)`, and its intensity is a parameter in `(0, oo)`, one over the temperature of its thin subclass in physics.
- `DriveLaw(target, function)` changes a whole lattice state by `function(values, *, random, **parameters)`.
- `PointsVisual(position, color_by, color_range, trail)` and `ImageVisual` in `skcomplex.engine.viz`.
- `EntropyAnalysis` and `DistributionAnalysis` in `skcomplex.analysis`, and `register_model(field=, observables=)` in
  `skcomplex.models`.
- The errors of `skcomplex.errors`, which hold no error for a missing extra yet.

## Goals / Non-Goals

**Goals:**
- Fit the three models into `System`, `Simulation`, and `Trajectory` without a new runtime: walks and folding are Monte
  Carlo time, and ensembles are replayed one member per frame.
- Keep the core free of molecular biology. A row score is any function of a row, a chain is any sequence of points, and
  a replay is any drive that reads its step.
- Take parsing, internal coordinates, superposition, hydrophobicity scales, RNA folding, and trajectory formats from
  mature libraries.

**Non-Goals:**
- Molecular dynamics with force fields, which `LammpsSolver` covers.
- Tertiary structure prediction, DSSP, RNA partition functions, and RNA design.
- Population genetics on genotype networks.

## Decisions

### Library survey

Checked on 2026-10-19 against PyPI, GitHub, and spike installs on macOS arm64 with Python 3.13:

| Library | Licence | Role | Decision |
|---|---|---|---|
| Biopython 1.88 | permissive, partly BSD-3-Clause | PDB and mmCIF, internal coordinates, superposition, hydrophobicity scales | adopt behind the extra `structures`: no macOS x86_64 wheels |
| ViennaRNA 2.7.2 | free for research and commercial use, no redistribution for a fee, not OSI approved | MFE folding, base-pair distance, plane layouts | adopt behind the extra `rna` |
| MDTraj 1.11.2 | LGPL-2.1-or-later | DCD, XTC, TRR, NetCDF, and HDF5 trajectories | adopt behind the extra `trajectories` |
| MDAnalysis 2.11.0 | LGPL-3.0-or-later | trajectories | reject: heavier than MDTraj and under a stricter licence |
| gemmi 0.7.5 | MPL-2.0 | fast mmCIF and PDB I/O | reject: Biopython also gives internal coordinates and superposition |
| seqfold 0.10.2 | MIT | MFE folding | reject: its folds differ from ViennaRNA's, which the literature on neutral networks uses |
| PeptideBuilder 1.1 | MIT | peptides from dihedrals | reject: inactive since 2020, Biopython does the same |
| SciPy | BSD-3-Clause | `Rotation.align_vectors`, bounded minimisation | already a dependency |
| NetworkX | BSD-3-Clause | neutral networks and their components | already a dependency |

Biopython is permissive and needs only NumPy, but the audit of 2026-11-01 found that 1.88 publishes wheels for Python
3.12 and 3.13 only for macOS arm64 and manylinux x86_64 and aarch64, with no macOS x86_64 wheel, so an Intel Mac builds
it from source with a C compiler. Only the two protein models of `skcomplex.models.biology.systems_biology` use it, and
the Project Profile lets an optional extra hold a library that a model alone needs, so Biopython lives behind the extra
`structures` rather than in the default install, which therefore installs everywhere from wheels. The documentation of
the extra states that it builds from source on Intel Macs. ViennaRNA's licence cannot be promised to the users of an
MIT package, and MDTraj is LGPL, so both live behind extras too.

The audit also corrected the reason MDAnalysis was rejected: 2.11.0, of 2026-10-21, now ships manylinux aarch64 wheels
for Python 3.12 and 3.13. MDTraj stays the choice, since it is lighter, reads the same trajectory formats the models
need, and is LGPL-2.1-or-later where MDAnalysis is LGPL-3.0-or-later. A model that needs a missing extra raises
`MissingExtraError`, which names the model, the extra, and the install command. The import is lazy with a suppression
that gives the reason, as the constitution asks.

The spike measured that ViennaRNA folds `GGGGAAACCCC` into `((((...))))` at −4.5 kcal/mol, that a neutral walk of 8
walkers of length 30 for 300 steps with robustness and evolvability every 10 steps takes 3.0 s, with a Spearman
correlation of robustness and evolvability between −0.69 and −0.77, and that Biopython parses the 38 models of 1L2Y in
0.18 s with an `RD` between 0.69 and 0.85.

### How the models fit the runtime

| Model | A unit of time | Core parts |
|---|---|---|
| `rna_neutral_networks` | one proposed point mutation per walker | `SamplingInteraction` with a function move and a `RowScoreSpec` |
| `fuzzy_oil_drop`, replay | one member of the ensemble | `DriveLaw` reading `time` |
| `fuzzy_oil_drop`, folding | one sweep of dihedral moves | `SamplingInteraction` with a rotation move and a `RowScoreSpec` |
| `early_stage_folding_intermediate` | one heat-bath sweep of path cells | `SamplingInteraction` with the `'heat-bath'` rule and a local energy |

Every model runs under `DiscreteTime()`.

### Row scores

`RowScoreSpec(function, proposals=1)` is a frozen spec of `skcomplex.specs` whose `function(row, **parameters)` takes
the values of one row of a lattice state, the last axis, and the parameters of the target that it names, and returns
one number. `SamplingInteraction` takes it as its `objective` and runs it as a private method of its own: each step it
proposes `proposals` moves per row through its move, evaluates the function on the proposed row, and accepts by its
rule on the difference. The rows are independent copies, each with its own draws from the simulation's stream. The
score of every row is recorded as the observable `<label>.score`.

The intensity domain of `SamplingInteraction` becomes `(0, oo]`. At an infinite intensity, the zero temperature of its
thin subclass in physics, the Metropolis rule accepts a move only when the objective does not grow, which gives neutral walks, with a score of 0 on the target phenotype and 1 off
it, and adaptive walks, with the base-pair distance. A `RowScoreSpec` whose function is not callable, or whose target is
not on a `LatticeSpace`, raises a `ValidationError` that the compiler reports with the existing code `law-space` or
`value`.

### Drives that read the step

`DriveLaw` passes `time`, the time of the step, to a function that names it in its signature, as it passes the
parameters it names. A replay is then a model function, `lambda values, *, time: ensemble[min(int(time), last)]`, which
holds the last member after the end. No replay law is added.

### Chain geometry

`ShapeAnalysis(kind, group=None, reference=None)` reads the positions of the members of a group from a trajectory, or an
array of frames by points by dimensions:
- `'gyration'` sets `values_`, the radius of gyration per frame.
- `'end-to-end'` sets `values_`, the distance between the first and last point per frame.
- `'rmsd'` sets `values_`, the root mean square deviation from `reference` after optimal superposition with SciPy's
  `Rotation.align_vectors`.

### Visualization

`PointsVisual(strip=True)` logs the positions of a group as one line strip in member order under `<path>/strip`, in two
or three dimensions, beside the points. Sequences use the existing `ImageVisual` with one color per label.

### Where the biology lives

The two protein models share the package `skcomplex.models.biology.systems_biology`, so their realizations live there,
and the RNA pieces live in `skcomplex.models.biology.evolution`:
- `systems_biology` holds the structures, the geometry, and the analysis of the two protein models:
  - `read_structure` and `write_structure` over Biopython's PDB and mmCIF parsers and writers with the extra
    `structures`, which return a frozen `StructureSpec` with coordinates of shape `(models, atoms, 3)` and the atom,
    residue, and chain names, and `read_trajectory` over MDTraj with the extra `trajectories`.
  - The protein backbone geometry, which writes `phi`, `psi`, and `omega` into Biopython's `IC_Chain` and rebuilds the
    atoms, and effective atoms, the mean atom of each residue.
  - `FuzzyOilDropAnalysis(BaseAnalysis)` with the kinds `'profile'`, the distributions `T`, `O`, and `R`,
    `'relative-distance'`, and `'environment-factor'`, after Eqs. 1.1–1.9.
  - The ellipse path of Brylinski et al. (2004, Bioinformatics 20:199) with its projection.
  - `BackboneVisual(PointsVisual)`, which rebuilds the backbone from the dihedral states of every frame.
  - The bundled entries, gzipped mmCIF files read with `importlib.resources`.
- `evolution`: the fold of a sequence through `RNA.fold_compound`, memoised per sequence, the base-pair distance, and
  the plane layout, and `NeutralNetworkAnalysis(BaseAnalysis)` with the kinds `'network'`, `'genotype-robustness'`,
  `'genotype-evolvability'`, `'phenotype-robustness'`, and `'phenotype-evolvability'`, after Wagner's definitions as
  Raman (2021 §13.3) states them, over any function from a sequence to a phenotype key.

The fuzzy oil drop takes each `σ` as a third of the largest absolute coordinate along its principal axis, after the
three-sigma rule of the book, and the Kyte-Doolittle scale of Biopython rescaled to `[0, 1]` by default. `K` minimises
`D_KL(O|M)` over `[0, 5]` with SciPy's bounded scalar minimiser. The early-stage model samples the cells of the path,
72 by default, with the heat-bath rule and the local energy `-T log p(cell | amino acid)`, whose probabilities come from
the projected native angles of the bundled structures with add-one smoothing. The section Source check says which of
these come from a source.

### Library models

- `rna_neutral_networks` (`biology/evolution`): a `LatticeElement` of walkers on `LatticeSpace((walkers, length))`, and
  a `SamplingInteraction` with the point mutation move, a `RowScoreSpec` that folds through ViennaRNA, and temperature
  zero. It raises `MissingExtraError` without the extra `rna`.
- `fuzzy_oil_drop` (`biology/systems_biology`): replays 1L2Y by default through a `DriveLaw`, and with
  `dynamics='folding'` samples the dihedral angles with a `RowScoreSpec` of `RD` and a clash penalty. Its catalog
  entry declares `RD` and `K` as observables.
- `early_stage_folding_intermediate` (`biology/systems_biology`): samples the path cells of BPTI's residues.

Both protein models read the bundled structures through Biopython, so they raise `MissingExtraError` without the extra
`structures`, and their catalog entries stay listed without it.

All three stay within the gallery budget of 300 frames and 60 s.

## Source check

The models were checked on 2026-11-04 against the PDFs of the project. Brylinski et al. 2004 and 2005, Levitt 1976,
Lorenz et al. 2011, and Wagner 2008 are not among them, and neither book holds their text. The web gave no copy either:
the open copy of Brylinski et al. 2005 and other papers of the group on PubMed Central answered with a reCAPTCHA, and
the fetch of a third-party copy of Jurkowski et al. 2004 was blocked, so neither was pursued further.

### Ellipse path

- Konieczny et al. (2023 §3.7.1, pp. 149 to 150) prints no equation, value, sign, or direction of the ellipse. It draws
  the path in Fig. 3.43d and e, and it does not cite Brylinski et al. Eq. 3.1 is the fit `ln(R) = 3.4e-4 V^2 - 2.009e-2
  V + 0.848` of the curvature radius against the V-angle, not the ellipse, so the proposal cites it loosely.
- A least squares fit to the pixels of Fig. 3.43e, in the form of `EllipsePath`, gives `A = 84.5°`, `B = 56.3°`, a
  centre within 1° of the origin, and the diagonals along the axes within 1°. One pixel of the figure is 2.7°. The
  defaults `A = 83°` and `B = 58°` agree with it, and the long diagonal from the upper left to the lower right agrees
  with the signs of `EllipsePath`. So the defaults stay, as readings of a figure.
- The origin and the direction of `t` have no source at hand, and the statement that the form is Eq. A.2 of Brylinski et
  al. (2005) could not be checked, so the docstring no longer makes it.
- Task 4.3 stays open, since it asks for the parameters of Brylinski et al. 2004, which nobody has read.

### Fuzzy oil drop

- Eq. 1.2 (p. 38), read from the rendered page, is `1 - (7 (r/c)^2 - 9 (r/c)^4 + 5 (r/c)^6 - (r/c)^8) / 2`, with `c = 9
  Å`. The code holds these coefficients, Eq. 1.1, and Eqs. 1.3 to 1.6.
- Eq. 1.9 (p. 40) is `M_i = [3DG_i + K (T_max - T_i)_n]_n`, with the complement normalised before `K` multiplies it. The
  code left that inner normalisation out, and it now holds it. The peak of the curve of `K = 0.5` in Fig. 1.41 is about
  0.68 of the peak of `T`, which matches the 0.67 of the printed equation and not the 0.59 of the code before. `K` of
  the first models changes from 0.31 to 2.68 for 1L2Y, from 0.27 to 0.98 for 1UBQ, and from 0.61 to 2.19 for 5PTI.
- `σ` as a third of the largest coordinate is one reading of the book, which has the Gaussian reach near zero at `3σ`
  (p. 33) and draws the axes of Figs. 1.40 and 1.41 to `±3σ`. The book gives no formula for `σ`.
- These have no source: the principal axes as the axes of Eq. 1.1, the Kyte-Doolittle scale, for which the book names no
  scale, `K` as the minimiser of `D_KL(O|M)`, leaving a residue out of its own sum, and the upper bound 5 of `K`, for
  which the book only reports values above 3 for membrane proteins (p. 41). The sum of Eq. 1.2 prints no `j ≠ i`, while
  the text speaks of interactions between residues (p. 38), so the book does not settle the self term.
- 72 cells and add-one smoothing have no source. The 72 of the book counts the 5° steps of one angle of the whole map
  (p. 147). Column F of Table 3.3 reaches 6.63 bits for glycine at 5°, above the 6.17 bits that 72 cells can hold, so
  the cells of the book are not 72 equal cells of `t`. The book does not state their number, so the default stays.
- Not reproduced: the book expects fast-folding proteins near `K = 0` with `RD < 0.5`, and most single-chain proteins at
  `0 < K < 0.5` (pp. 39 to 40). The mean `RD` and `K` here are 0.77 and 2.2 for 1L2Y, 0.70 and 0.98 for 1UBQ, and 0.77
  and 2.2 for 5PTI. The models of 1L2Y range over `RD` 0.69 to 0.85 and `K` 1.5 to 3.2. The code is not at fault in any
  point the book states, the gap comes from `σ`, which it does not state, and `σ` stays as it is, since nothing at hand
  says what it should be. A new `σ` picked to reach the expected values would be a tuned number.
- The equations hold against the one worked example of the book. The curves `T` and `O` read from the pixels of Fig.
  1.41 give `K` of 0.49 to 0.50 by the minimiser of the code, against the printed 0.5, and a least squares fit of the
  teal curve to Eq. 1.9 gives 0.50 to 0.52. They give `RD` of 0.60 to 0.68 against the 0.700 of Fig. 1.40, a figure that
  depends on where the tails of `T` end, between 0 and 4 pixels above the axis. So the code reproduces the numbers of
  the figure, which also supports the minimiser, and the figures are drawings reduced to one dimension, which cannot
  calibrate `σ` for a protein.
- The book gives no `RD` or `K` of a named entry. It gives none for the Trp-cage, ubiquitin, or BPTI, and that 1L2Y
  counts as fast-folding comes from outside the book. Its PDB codes at this section, 6SDZ and 1DVQ (p. 47), carry no
  values, and Fig. 1.43 only says "K < 0.5" for an unnamed entry of alpha-synuclein. The book says that nearly all
  domains reach `K = 0` when analysed as standalone units (p. 40), which does not apply, since these three are whole
  small chains.
- One choice changed at a time, mean `RD` and `K` of 1L2Y, 1UBQ, and 5PTI, against 0.77 and 2.2, 0.70 and 0.98, and 0.77
  and 2.2, with `K` searched over `[0, 50]`. No change but the width of the Gaussian brings all three below `RD = 0.5`,
  or `K` to the range of the book:

  | Choice | 1L2Y | 1UBQ | 5PTI |
  | --- | --- | --- | --- |
  | effective atom of the heavy atoms | 0.83, 3.4 | 0.70, 0.98 | 0.80, 3.0 |
  | Cα in place of the effective atom, against p. 38 | 0.84, 3.5 | 0.71, 0.97 | 0.87, 3.7 |
  | axes of the file in place of the principal axes | 0.82, 2.6 | 0.75, 1.2 | 0.85, 3.7 |
  | x axis through the farthest pair, y likewise | 0.76, 2.0 | 0.65, 0.82 | 0.76, 2.1 |
  | keeping a residue in its own sum | 0.84, 3.9 | 0.66, 0.89 | 0.76, 2.4 |
  | Kyte-Doolittle shifted by 5 in place of 4.5 | 0.78, 2.2 | 0.73, 1.1 | 0.78, 2.2 |
  | Kyte-Doolittle clipped at zero | 0.69, 7.4 | 0.39, 0.28 | 0.69, 1.9 |
  | Hopp-Woods negated, rescaled to `[0, 1]` | 0.72, 1.6 | 0.74, 1.2 | 0.72, 1.5 |
  | Janin, rescaled to `[0, 1]` | 0.74, 1.9 | 0.83, 1.5 | 0.77, 2.2 |
  | the same hydrophobicity for every residue | 0.82, 2.4 | 0.92, 2.3 | 0.86, 2.3 |
  | `σ` times 1.25 | 0.54, 0.81 | 0.51, 0.45 | 0.63, 0.99 |
  | `σ` times 1.5 | 0.37, 0.27 | 0.41, 0.20 | 0.53, 0.48 |
  | `σ` times 2 | 0.32, 0.00 | 0.38, 0.00 | 0.46, 0.07 |
  | `σ` as the standard deviation of the coordinates | 0.32, 0.18 | 0.44, 0.31 | 0.55, 0.60 |
  | `σ` as a third of the largest coordinate plus the cutoff | 0.34, 0.00 | 0.36, 0.08 | 0.54, 0.49 |

- Other inputs ruled out. A rescale of the scale changes nothing, since `O` is normalised, and the shift of 4.5 is the
  smallest that keeps every value of the scale at zero or above: the raw scale gives negative `O` at 3 of 20, 26 of 76,
  and 25 of 58 residues, where Eq. 1.3 has no value. A Powell search over the twenty values of a scale, which is no
  proof of a bound, stops at `RD` 0.56 fitted to 1L2Y alone, and at a worst `RD` of 0.63 fitted to all three, with a
  single amino acid above zero. The bound 5 of `K` is not reached with the default scale, and the minimiser agrees with
  a grid over `[0, 50]` within 0.002. The log base cancels in `RD`.
- Why `σ` decides. With the farthest atom at `3σ`, `T` holds most of its weight at the centre and almost none on the
  rim, from 0.016 to 9.7 times the uniform share in the first models, while `O` only spans 0.07 to 2.5, since a residue
  on the rim still has neighbours within 9 Å. So the residues with `T` under a third of the uniform share, 11 of 20 in
  1L2Y, 24 of 76 in 1UBQ, and 22 of 58 in 5PTI, carry nearly all of `D_KL(O|T)`: 1.13 of 1.13, 0.50 of 0.53, and 0.71 of
  0.83 bits.
- Open. The tool HPHOB of the authors, which the book names (p. 2, `https://hphob.sano.science`), computes these
  parameters for any structure and was not consulted, since this check stayed with the PDFs. Its output for 1L2Y, 1UBQ,
  and 5PTI, or a paper of the group that defines `σ`, would settle `σ` and the scale.

### Folding

The book has no folding simulation, and neither "clash" nor a rotation width appears in it. The clash rule of under 3 Å,
at least 2 residues apart, and +1 per clash, the width of 60°, the extended start of `phi = -120°` and `psi = 130°`, the
temperature 0, and one proposal per unit of time are choices of this change, and the docstring says so.

### RNA neutral networks

- Raman (2021 §13.2 to §13.3, pp. 298 to 304) gives RNA of length 30, the `3L` neighbours, the four definitions of
  Wagner, and the correlations, and the code agrees with them with one question: the text counts "the number of
  different phenotypes found in the 1-neighbourhood", without saying whether the phenotype of the genotype itself
  counts. The spec scenario of the toy map asks for 1, which leaves it out, so the code stays and the docstring says so.
- Raman cites Wagner, Proc. R. Soc. B 275:91 (2008), as reference 17, which is the `Wagner 2008` of the model. It does
  not cite Lorenz et al. 2011 or ViennaRNA, and it cites Schuster et al. 1994 and Ancel and Fontana 2000 for RNA.
  Konieczny et al. name Levitt with the placeholder `[lit]` and no year, so `Levitt 1976` has no local support.

## Risks / Trade-offs

- [ViennaRNA's licence is not OSI approved] → It is never installed by default, and the extra's documentation states
  the licence. The tests, docs, and gallery sessions install it explicitly.
- [Row scores of Python functions are slow on long rows] → The memo removes repeated folds, and neutral walks stay at
  lengths where ViennaRNA folds in under a millisecond.
- [The book does not fix `σ`, the scale, or the ellipse parameters] → Each is a parameter whose default and source the
  docstring names, and the scenarios test properties that hold for any choice.
- [Biopython 1.88 publishes no macOS x86_64 wheels] → It sits behind the extra `structures`, so the default install
  never builds it. Installing the extra on an Intel Mac builds it from source, which the documentation of the extra
  states, and the CI jobs that install the extras run on platforms with wheels.
- [Biopython's internal coordinates warn on NumPy's `where` without `out`] → The package filters this one warning by
  category and module, and a test checks that no other warning is hidden.

## Migration Plan

Additive. Rollback removes the new names and the extras, and returns the three inventory entries to blocked.
