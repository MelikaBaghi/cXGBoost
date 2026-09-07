# cXGBoost

Constrained Extreme Gradient Boosting for adapting reduced-order models.

A proper orthogonal decomposition basis built at one parameter loses accuracy as
the parameter moves, so the basis has to be predicted at conditions that were
never simulated. A basis is a point on the Grassmann manifold, and existing
predictors regress it through a logarithmic mapping whose coordinate identifies a unique
subspace only inside a ball of radius pi/2. They enforce that bound after
fitting, or not at all.

cXGBoost imposes it during the fit. Every leaf of a vector-valued boosting
ensemble solves a quadratically constrained sub-problem which, under the squared
Euclidean loss, is exactly a Euclidean projection of the ordinary leaf value onto
an intersection of balls. That is available in closed form when one ball is
active and by Dykstra's algorithm otherwise, so no general-purpose optimiser is
called at any candidate split. The fitted ensemble satisfies the bound at every
training parameter and at every truncation, and no correction is applied to the
prediction.

## Use

```python
import numpy as np
from cxgboost import make_predictor

# mu: (N, d) parameters.  Phi: list of N bases, each (n, r) with orthonormal columns.
model = make_predictor("cxgboost", chart="exact").fit(mu, Phi)
Phi_star = model.predict(mu_query)      # predicted basis at an unseen parameter
y = model.predict_tangent(mu_query)     # its mapped coordinate, ||y|| <= pi/2
```

Pass `chart="pca"` when the full coordinate is too large to form, which is the case
whenever `n * r` is big. Pass `make_predictor("unconstrained")` for the
constraint-off variant used in the paper's ablation.

## Contents

- `cxgboost/grassmann.py` holds the exponential and logarithmic maps, the
  principal angles, and the projection error used as the reported metric
- `cxgboost/chart.py` builds the logarithmic mapping and its PCA variant
- `cxgboost/boosting.py` is the constrained ensemble and the leaf projection
- `cxgboost/predictor.py` is the `fit` and `predict` wrapper
- `paper/cXGBoost_JCP.pdf` is the manuscript

## Scope of this release

This repository carries the proposed estimator. The two comparison methods in
the paper, Grassmann interpolation and the projected Gaussian process, are our
reimplementations of published work and are not redistributed here; see the
references in the manuscript for the original sources.

## Citation

If you use this code, please cite the manuscript in `paper/`.

## Licence

MIT. See `LICENSE`.

## Requirements

Python 3.9 or later and NumPy. SciPy is used for the Gaussian-process baseline
only and is not needed here.
