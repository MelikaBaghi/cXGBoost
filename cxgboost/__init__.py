"""Constrained Extreme Gradient Boosting for adapting reduced-order bases.

Predicting a parameter-dependent POD basis is regression whose output lives on
the Grassmann manifold. Existing predictors regress through a logarithmic
mapping whose coordinate identifies a unique subspace only inside a ball of
radius pi/2, and enforce that bound, if at all, after fitting. cXGBoost imposes it inside the
fit: every leaf of a vector-valued boosting ensemble solves a quadratically
constrained sub-problem, which under the squared Euclidean loss is a Euclidean
projection onto an intersection of balls.

    from cxgboost import make_predictor
    model = make_predictor("cxgboost", chart="exact").fit(mu_train, Phi_train)
    Phi_star = model.predict(mu_query)

This release carries the proposed estimator only. The comparison methods in the
paper are the authors' reimplementations of published work and are not
redistributed.
"""

from .boosting import CXGBoost
from .chart import Chart, PCAChart, build_chart
from .grassmann import (PI_2, clip_to_ball, exp_map, geodesic_distance, log_map,
                        orthonormalize, principal_angles, projection_error)
from .predictor import CXGBoostSubspaceRegressor, SubspacePredictor, make_predictor

__all__ = [
    "CXGBoost", "CXGBoostSubspaceRegressor", "SubspacePredictor",
    "make_predictor", "Chart", "PCAChart", "build_chart",
    "exp_map", "log_map", "principal_angles", "geodesic_distance",
    "projection_error", "orthonormalize", "clip_to_ball", "PI_2",
]
