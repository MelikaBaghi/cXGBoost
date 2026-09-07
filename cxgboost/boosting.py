"""cXGBoost: gradient boosting with vector-valued leaves constrained to a ball.

Ordinary boosting fits scalar leaves; here every leaf emits a vector ``w`` in
chart coordinates and the running prediction must stay inside the ball that
makes the Grassmann exponential map injective. Enforcing that *during* training
(rather than projecting afterwards) is the method's contribution.

Leaf subproblem, for a leaf holding samples ``I`` with running predictions
``y_i`` and gradients ``g_i``::

    minimise    G . w + 0.5 * H * ||w||^2          G = sum_i g_i,  H = |I| + lambda
    subject to  ||y_i + w|| <= rho                 for all i in I

The Hessian of the squared loss is isotropic, so the objective is a plain
strictly convex quadratic and the feasible set is an intersection of equal-radius
balls: a unique minimiser always exists, and ``w = 0`` is feasible whenever the
current iterate is (which is how feasibility propagates across rounds).

Because the Hessian is isotropic the objective is ``0.5 * H * ||w - w*||^2`` up to
a constant, with ``w* = -G / H`` the unconstrained step. So **the leaf subproblem
is exactly the Euclidean projection of the unconstrained step onto an
intersection of balls** -- which is what makes the fast solvers below exact
rather than heuristic.

Solvers (``constraint_solver``):

``"auto"`` (default)
    ``w*`` if feasible; else the closed-form radial projection onto a single
    violated ball when that already satisfies the rest (the exact optimum by
    KKT); else Dykstra's alternating projection onto the intersection, which
    converges to the same projection. No external solver, exact to tolerance.
``"shrink"``
    Scale ``w*`` toward zero until every ball is satisfied. Always feasible,
    generally *not* optimal -- kept because it is what the original notebooks
    did, so published numbers can be reproduced.
``"qcqp"``
    Exact, via cvxpy. Orders of magnitude slower; used to validate the others
    (see ``tests/test_boosting.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from numpy.linalg import norm


def _nrm(v: np.ndarray) -> float:
    """Euclidean norm of a 1-D array.

    ``numpy.linalg.norm`` spends most of its time in dispatch when the array is
    short, and the leaf solve calls it hundreds of thousands of times on vectors
    of a few hundred entries. The dot product is the same arithmetic without
    that overhead.
    """
    return sqrt(float(v @ v))


def _within(centres: np.ndarray, w: np.ndarray, rho: float) -> bool:
    """Whether ``||c + w|| <= rho`` for every centre, without forming the norms."""
    d = centres + w
    return bool(np.all(np.einsum("ij,ij->i", d, d) <= rho * rho + 1e-9))

from .grassmann import PI_2, RHO

SOLVERS = ("auto", "shrink", "qcqp")


@dataclass
class _Node:
    weight: np.ndarray
    slope: np.ndarray | None = None      # (m, d) local linear term, linear leaves
    centre: np.ndarray | None = None     # (d,) point the linear term expands about
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class FitStats:
    """Diagnostics collected during :meth:`CXGBoost.fit`."""

    mse: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    n_leaves: int = 0
    n_active_leaves: int = 0
    n_solver_fallbacks: int = 0
    #: Fallbacks incurred at *accepted* nodes only. ``n_solver_fallbacks``
    #: counts every solver call, including the two per candidate split inside
    #: the feature/threshold search, which outnumber accepted nodes by more than
    #: an order of magnitude. Dividing that total by ``n_active_leaves``, which
    #: counts accepted nodes alone, mixes two populations and understates the
    #: denominator. This counter shares the denominator's population.
    n_accepted_fallbacks: int = 0

    @property
    def active_fraction(self) -> float:
        """Share of leaves where the ball constraint actually bound."""
        return self.n_active_leaves / max(self.n_leaves, 1)


class CXGBoost:
    """Constrained multivariate gradient boosting.

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, min_leaf, n_splits, gamma, reg_lambda
        Usual boosting knobs. ``gamma`` is the minimum gain to accept a split,
        ``reg_lambda`` is the ridge term added to the leaf Hessian.
    constrained
        Turn the ball constraint off to get the ablation baseline.
    constraint_radius, constraint_offset
        The ball is ``||y_i + offset + w|| <= radius``. A chart that centres its
        coordinates reports a non-zero offset so the constraint still refers to
        ``||Z||_F``; see :meth:`cxgboost.chart.Chart.tangent_norm`.
    feature_subsample
        Fraction of features considered per node.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.2,
        max_depth: int = 2,
        min_leaf: int = 2,
        n_splits: int = 20,
        gamma: float = 1e-3,
        reg_lambda: float = 1e-2,
        constrained: bool = True,
        constraint_radius: float = RHO,
        constraint_offset: np.ndarray | None = None,
        constraint_solver: str = "auto",
        leaf_model: str = "constant",
        leaf_ridge: float = 1e-6,
        feature_subsample: float = 1.0,
        random_state: int | None = 0,
        verbose: bool = False,
    ):
        if constraint_solver not in SOLVERS:
            raise ValueError(f"constraint_solver must be one of {SOLVERS}")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_splits = n_splits
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.constrained = constrained
        self.constraint_radius = constraint_radius
        self.constraint_offset = constraint_offset
        self.constraint_solver = constraint_solver
        if leaf_model not in ("constant", "linear"):
            raise ValueError("leaf_model must be 'constant' or 'linear'")
        self.leaf_model = leaf_model
        self.leaf_ridge = leaf_ridge
        self.feature_subsample = feature_subsample
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ leaves

    def _leaf_weight(self, G: np.ndarray, n: int, y_prev: np.ndarray):
        """Solve the leaf subproblem. Returns ``(w, constraint_was_active)``."""
        H = n + self.reg_lambda
        w = -G / H
        if not self.constrained:
            return w, False

        centres = y_prev if self.constraint_offset is None else y_prev + self.constraint_offset
        rho = self.constraint_radius

        d0 = centres + w
        sq = np.einsum("ij,ij->i", d0, d0)
        violated = np.where(sq > rho * rho)[0]
        if violated.size == 0:
            return w, False

        if self.constraint_solver == "shrink":
            return self._shrink(w, centres[violated], rho), True

        if self.constraint_solver == "qcqp":
            return self._qcqp(G, H, centres, rho, w), True

        # "auto": try the closed form for a single active ball, then Dykstra.
        for i in violated:
            cand = self._project_onto_ball(w, centres[i], rho)
            if _within(centres, cand, rho):
                return cand, True

        # Active set: alternate over the balls that are actually violated, then
        # verify against all of them. A ball the iterate already satisfies
        # contributes nothing to the projection, and dropping it from the sweep
        # is what makes the cost scale with the active set rather than the leaf
        # size. If the verification fails, the newly violated balls are added
        # and the sweep is repeated, so the returned point is checked against
        # every constraint either way.
        active = list(violated)
        for _ in range(len(centres) + 1):
            cand = self._dykstra(w, centres[active], rho)
            d = centres + cand
            sq = np.einsum("ij,ij->i", d, d)
            bad = np.where(sq > rho * rho + 1e-9)[0]
            if bad.size == 0:
                return cand, True
            new = [int(i) for i in bad if i not in active]
            if not new:
                break
            active.extend(new)
        self._stats.n_solver_fallbacks += 1
        return self._shrink(cand, centres, rho), True

    @classmethod
    def _dykstra(cls, w, centres, rho, max_iter: int = 200, tol: float = 1e-10):
        """Dykstra's alternating projection onto the intersection of the balls.

        Converges to the Euclidean projection of ``w`` (unlike plain cyclic
        projection, which only finds *some* point in the intersection).
        """
        m = centres.shape[0]
        x = w.copy()
        corr = np.zeros((m, w.shape[0]))
        prev = np.empty_like(x)
        rho2 = rho * rho
        for _ in range(max_iter):
            prev[:] = x
            for i in range(m):
                ci = centres[i]
                y = x - corr[i]
                d = y + ci
                s2 = float(d @ d)
                if s2 <= rho2 or s2 < 1e-60:
                    x_new = y
                else:
                    x_new = d * (rho / sqrt(s2)) - ci
                corr[i] = x_new - y
                x = x_new
            diff = x - prev
            if _nrm(diff) <= tol * (1.0 + _nrm(x)):
                break
        return x

    @staticmethod
    def _project_onto_ball(w: np.ndarray, centre: np.ndarray, rho: float) -> np.ndarray:
        """Euclidean projection of ``w`` onto ``{v : ||centre + v|| <= rho}``."""
        d = w + centre
        s = _nrm(d)
        if s <= rho or s < 1e-30:
            return w
        return d * (rho / s) - centre

    @staticmethod
    def _shrink(w: np.ndarray, centres: np.ndarray, rho: float) -> np.ndarray:
        """Largest ``s`` in [0, 1] with ``||c + s w|| <= rho`` for every centre."""
        ww = float(w @ w)
        if ww < 1e-30:
            return w
        s = 1.0
        for c in centres:
            b = 2.0 * float(c @ w)
            k = float(c @ c) - rho * rho
            disc = b * b - 4.0 * ww * k
            si = 0.0 if disc < 0 else (-b + np.sqrt(disc)) / (2.0 * ww)
            s = min(s, max(0.0, si))
        return s * w

    def _qcqp(self, G, H, centres, rho, w_unconstrained):
        try:
            import cvxpy as cp
        except ImportError:  # pragma: no cover - optional dependency
            self._stats.n_solver_fallbacks += 1
            return self._shrink(w_unconstrained, centres, rho)
        w = cp.Variable(G.shape[0])
        prob = cp.Problem(
            cp.Minimize(G @ w + 0.5 * H * cp.sum_squares(w)),
            [cp.norm(centres[i] + w, 2) <= rho for i in range(centres.shape[0])],
        )
        try:
            prob.solve()
        except Exception:
            prob.status = "failed"
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.asarray(w.value, dtype=float)
        self._stats.n_solver_fallbacks += 1
        return self._shrink(w_unconstrained, centres, rho)

    # ------------------------------------------------------------------- trees

    @staticmethod
    def _objective(G, H, w) -> float:
        return float(G @ w) + 0.5 * H * float(w @ w)

    def _split_candidates(self, values: np.ndarray) -> np.ndarray:
        uniq = np.unique(values)
        if uniq.size < 2:
            return np.empty(0)
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        if mids.size > self.n_splits:
            idx = np.linspace(0, mids.size - 1, self.n_splits).astype(int)
            mids = mids[idx]
        return mids

    def _fit_linear_leaf(self, X, g, y_prev, idx):
        """Leaf value ``w + V (theta - theta_bar)``, with the constraint intact.

        The slope is an unconstrained ridge least-squares fit of the leaf's
        residuals against the centred parameters. The intercept is then the
        solution of exactly the sub-problem of :meth:`_leaf_weight`, because
        requiring ``||c_i + w + V d_i|| <= rho`` for every sample in the leaf is
        the same intersection-of-balls problem with each centre shifted by
        ``V d_i``. The projection identity, its closed form, the Dykstra branch
        and the shrinkage argument therefore carry over unchanged; only the ball
        centres move.
        """
        G = g[idx].sum(axis=0)
        theta_bar = X[idx].mean(axis=0)
        delta = X[idx] - theta_bar                      # (n, d), sums to zero
        R = -g[idx]                                     # (n, m) desired outputs
        S = delta.T @ delta + self.leaf_ridge * np.eye(delta.shape[1])
        V = np.linalg.solve(S, delta.T @ R).T           # (m, d)
        shift = delta @ V.T                             # (n, m)
        w, active = self._leaf_weight(G, len(idx), y_prev[idx] + shift)
        return w, V, theta_bar, active

    def _build(self, X, g, y_prev, idx, depth) -> _Node:
        G = g[idx].sum(axis=0)
        _fb_before = self._stats.n_solver_fallbacks
        w, active = self._leaf_weight(G, len(idx), y_prev[idx])
        self._stats.n_leaves += 1
        self._stats.n_active_leaves += int(active)
        self._stats.n_accepted_fallbacks += (
            self._stats.n_solver_fallbacks - _fb_before)
        node = _Node(weight=w, n_samples=len(idx))

        if depth <= 0 or len(idx) < 2 * self.min_leaf:
            return self._finalise(node, X, g, y_prev, idx)

        parent = self._objective(G, len(idx) + self.reg_lambda, w)
        best_gain, best = self.gamma, None
        for f in self._features:
            col = X[idx, f]
            for thr in self._split_candidates(col):
                left = idx[col <= thr]
                right = idx[col > thr]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                Gl, Gr = g[left].sum(axis=0), g[right].sum(axis=0)
                wl, _ = self._leaf_weight(Gl, len(left), y_prev[left])
                wr, _ = self._leaf_weight(Gr, len(right), y_prev[right])
                gain = (
                    parent
                    - self._objective(Gl, len(left) + self.reg_lambda, wl)
                    - self._objective(Gr, len(right) + self.reg_lambda, wr)
                )
                if gain > best_gain:
                    best_gain, best = gain, (f, thr, left, right)

        if best is None:
            return self._finalise(node, X, g, y_prev, idx)

        f, thr, left, right = best
        node.feature, node.threshold = f, float(thr)
        node.left = self._build(X, g, y_prev, left, depth - 1)
        node.right = self._build(X, g, y_prev, right, depth - 1)
        return node

    def _finalise(self, node, X, g, y_prev, idx):
        """Refit a terminal node as a linear leaf, if that mode is on."""
        if self.leaf_model != "linear" or len(idx) < 2:
            return node
        w, V, centre, active = self._fit_linear_leaf(X, g, y_prev, idx)
        node.weight, node.slope, node.centre = w, V, centre
        return node

    @staticmethod
    def _tree_predict(node: _Node, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], node.weight.shape[0]))
        for i, x in enumerate(X):
            nd = node
            while not nd.is_leaf:
                nd = nd.left if x[nd.feature] <= nd.threshold else nd.right
            out[i] = (nd.weight if nd.slope is None
                      else nd.weight + nd.slope @ (x - nd.centre))
        return out

    # --------------------------------------------------------------------- API

    def fit(self, X, Y) -> "CXGBoost":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] != len(Y):
            X = X.T if X.T.shape[0] == len(Y) else X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        k = max(1, int(round(self.feature_subsample * n_features)))
        self._features = (
            np.arange(n_features)
            if k >= n_features
            else rng.choice(n_features, size=k, replace=False)
        )

        self._stats = FitStats()
        self.init_ = Y.mean(axis=0)
        self.base_score_projected_ = False

        if self.constrained:
            # The feasibility induction (Prop. 1) starts from the base score, so
            # the base score itself has to satisfy the enforced ball -- every
            # later iterate is built on it. The target mean is feasible by
            # convexity whenever the targets are, but that is sufficient and not
            # necessary, and a target outside the ball can carry the mean out
            # with it. The condition is therefore enforced here rather than
            # assumed, by projecting the base score onto the same ball the leaf
            # updates use. When the mean is already feasible, which is the case
            # on every benchmark reported, the projection is the identity and
            # nothing about the fit changes.
            shift = (0.0 if self.constraint_offset is None
                     else self.constraint_offset)
            centre = self.init_ + shift
            b0 = float(norm(centre))
            if b0 > self.constraint_radius:
                self.init_ = centre * (self.constraint_radius / b0) - shift
                self.base_score_projected_ = True
            self.base_score_norm_ = b0

            # A target outside the ball is unreachable under the constraint --
            # the model will silently underfit it. Surface that separately.
            centres = Y if self.constraint_offset is None else Y + self.constraint_offset
            outside = int(np.sum(norm(centres, axis=1) > self.constraint_radius + 1e-9))
            self.n_infeasible_targets_ = outside
            if outside:
                import warnings

                warnings.warn(
                    f"{outside}/{len(Y)} training targets lie outside the "
                    f"constraint ball (radius {self.constraint_radius:.3f}); the "
                    "constrained model cannot fit them exactly. Consider a "
                    "reference closer to the data (reference='centroid') or check "
                    "whether the Frobenius bound is too conservative for this r.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            self.n_infeasible_targets_ = 0
            self.base_score_norm_ = float(norm(self.init_))

        self.trees_ = []
        y_prev = np.tile(self.init_, (X.shape[0], 1))

        for it in range(self.n_estimators):
            g = -(Y - y_prev)  # gradient of 0.5 * ||y - f||^2
            self._stats.grad_norm.append(float(norm(g, axis=1).mean()))
            tree = self._build(X, g, y_prev, np.arange(X.shape[0]), self.max_depth)
            self.trees_.append(tree)
            y_prev = y_prev + self.learning_rate * self._tree_predict(tree, X)
            self._stats.mse.append(float(np.mean((Y - y_prev) ** 2)))
            if self.verbose:
                print(
                    f"  round {it + 1:3d}/{self.n_estimators}  "
                    f"mse={self._stats.mse[-1]:.3e}  "
                    f"grad={self._stats.grad_norm[-1]:.3e}"
                )
            if self._stats.grad_norm[-1] < 1e-10:
                break

        self.stats_ = self._stats
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1) if len(self._features) == 1 else X.reshape(1, -1)
        y = np.tile(self.init_, (X.shape[0], 1))
        for tree in self.trees_:
            y = y + self.learning_rate * self._tree_predict(tree, X)
        return y

    def staged_predict(self, X):
        """Yield the prediction after each boosting round (for convergence plots)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.tile(self.init_, (X.shape[0], 1))
        for tree in self.trees_:
            y = y + self.learning_rate * self._tree_predict(tree, X)
            yield y.copy()
