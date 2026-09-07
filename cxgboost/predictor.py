"""End-to-end subspace regressors: parameter ``mu`` in, POD basis ``Phi`` out.

Every method here follows the same three steps, and differs only in the middle
one:

1. lift the training bases into the tangent space at a reference ``Phi0``
   (:func:`cxgboost.grassmann.log_map`) and flatten them through a chart;
2. regress the chart coordinates against ``mu``;
3. push the prediction back with the exponential map.

Step 3 is only well defined when the prediction is inside the injectivity ball,
which is exactly what :class:`CXGBoostSubspaceRegressor` guarantees and what the
interpolation baseline does not.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from .boosting import CXGBoost
from .chart import Chart, build_chart
from .grassmann import PI_2, RHO, clip_to_ball, exp_map, log_map


class SubspacePredictor:
    """Common interface: ``fit(mu, Phi_list)`` then ``predict(mu) -> Phi``."""

    chart: Chart
    Phi0: np.ndarray

    def fit(self, mu, Phi_list) -> "SubspacePredictor":
        raise NotImplementedError

    def predict_tangent(self, mu) -> np.ndarray:
        raise NotImplementedError

    def predict(self, mu) -> np.ndarray:
        Z = self.predict_tangent(mu)
        if self.safeguard:
            Z = clip_to_ball(Z, self.radius)
        return exp_map(Z, self.Phi0)

    # -- shared setup ------------------------------------------------------

    def _prepare(self, mu, Phi_list):
        mu = np.atleast_2d(np.asarray(mu, dtype=float))
        if mu.shape[0] != len(Phi_list):
            mu = mu.T
        self.mu_train_ = mu
        self.Phi0 = self._reference(Phi_list)
        self.Z_train_ = [log_map(P, self.Phi0, variant=self.log_variant) for P in Phi_list]
        return mu

    def _reference(self, Phi_list) -> np.ndarray:
        """Pick ``Phi0``. ``"middle"`` is the convention the paper's runs used."""
        ref = self.reference
        if isinstance(ref, (int, np.integer)):
            return np.array(Phi_list[int(ref)], dtype=float)
        if ref == "middle":
            return np.array(Phi_list[len(Phi_list) // 2], dtype=float)
        if ref == "centroid":
            # basis whose mean geodesic distance to the others is smallest
            from .grassmann import geodesic_distance

            d = [np.mean([geodesic_distance(P, Q) for Q in Phi_list]) for P in Phi_list]
            return np.array(Phi_list[int(np.argmin(d))], dtype=float)
        raise ValueError(f"unknown reference {ref!r}")

    def tangent_diagnostics(self) -> dict:
        """How close the *training* targets sit to the injectivity boundary."""
        f = np.array([norm(Z, "fro") for Z in self.Z_train_])
        s = np.array([norm(Z, 2) for Z in self.Z_train_])
        return {
            "frobenius_max": float(f.max()),
            "spectral_max": float(s.max()),
            # squared comparison with a roundoff tolerance: a value sitting on
            # the bound to floating point is inside, not outside
            "n_outside_frobenius": int((f * f > PI_2 * PI_2 + 1e-9).sum()),
            "n_outside_spectral": int((s * s > PI_2 * PI_2 + 1e-9).sum()),
        }


class CXGBoostSubspaceRegressor(SubspacePredictor):
    """The proposed method: constrained boosting in chart coordinates.

    ``constrained=False`` gives the ablation baseline (plain multivariate
    boosting); combine it with ``safeguard=True`` for the "unconstrained +
    post-hoc projection" variant reported in the ablation.
    """

    def __init__(
        self,
        chart: str = "auto",
        chart_dim: int = 20,
        chart_center: bool = True,
        reference="middle",
        log_variant: str = "projected",
        radius: float = RHO,
        constrained: bool = True,
        safeguard: bool = True,
        **boost_kwargs,
    ):
        self.chart_kind = chart
        self.chart_dim = chart_dim
        self.chart_center = chart_center
        self.reference = reference
        self.log_variant = log_variant
        self.radius = radius
        self.constrained = constrained
        self.safeguard = safeguard
        self.boost_kwargs = boost_kwargs

    def fit(self, mu, Phi_list):
        mu = self._prepare(mu, Phi_list)
        self.chart = build_chart(
            self.Phi0,
            self.Z_train_,
            kind=self.chart_kind,
            max_dim=self.chart_dim,
            center=self.chart_center,
        )
        y = self.chart.encode_all(self.Z_train_)
        offset, rho = self.chart.ball_params(self.radius)
        self.model_ = CXGBoost(
            constrained=self.constrained,
            constraint_radius=rho,
            constraint_offset=offset,
            **self.boost_kwargs,
        ).fit(mu, y)
        return self

    def predict_tangent(self, mu) -> np.ndarray:
        mu = np.atleast_2d(np.asarray(mu, dtype=float))
        y = self.model_.predict(mu)[0]
        return self.chart.decode(y)

    @property
    def stats(self):
        return self.model_.stats_



def make_predictor(kind: str = "cxgboost", **kwargs) -> SubspacePredictor:
    """Construct the constrained or unconstrained boosting regressor.

    This public release carries the proposed estimator only. The competing
    predictors evaluated in the paper, Grassmann interpolation and the projected
    Gaussian process, are the authors' reimplementations of published methods
    and are not redistributed here.
    """
    if kind in ("cxgboost", "constrained"):
        return CXGBoostSubspaceRegressor(constrained=True, **kwargs)
    if kind in ("unconstrained", "xgboost"):
        return CXGBoostSubspaceRegressor(constrained=False, **kwargs)
    raise ValueError(f"unknown predictor kind: {kind!r}")
