"""Charts: flatten a tangent vector ``Z`` (n x r) into a vector ``y`` the trees regress.

The boosting constraint is stated as ``||y|| <= pi/2``, so a chart is only sound
if it lets us recover ``||Z||`` from ``y``. Both charts here expose
:meth:`Chart.tangent_norm`, which returns the exact Frobenius norm of the decoded
tangent vector without ever materialising it.

Two implementations:

``ExactChart``
    The paper's ``F`` matrix (Eq. 6), dimension ``m = nr - r``. Isometric, so
    ``||y|| == ||Z||_F``. Needs an ``(nr - r) x nr`` dense matrix -- fine for the
    cylinder (895 x 900), impossible for the wave (259 200 x 259 210 ~ 538 GB).

``PCAChart``
    PCA over the training tangent vectors. With ``N`` training points at most
    ``N - 1`` directions carry variance, so ``m <= N - 1`` and the encoding is
    lossless *on the training set*. This is what every large benchmark uses.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd

from .grassmann import horizontal_basis


class Chart:
    """Interface: an isometry-aware linear map between ``Z`` and ``y``."""

    dim: int
    shape: tuple[int, int]

    def encode(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def tangent_norm(self, y: np.ndarray) -> float:
        """``||decode(y)||_F`` without building ``Z``."""
        raise NotImplementedError

    def ball_params(self, radius: float) -> tuple[np.ndarray, float]:
        """Recast ``||decode(y)||_F <= radius`` as a ball ``||y + b|| <= rho`` in y-space.

        The boosting solver only knows how to handle balls, so every chart has to
        report the ``(offset, effective_radius)`` pair that makes the constraint it
        enforces equal to the constraint we actually want.
        """
        raise NotImplementedError

    def encode_all(self, Z_list) -> np.ndarray:
        return np.vstack([self.encode(Z) for Z in Z_list])

    def roundtrip_error(self, Z_list) -> float:
        """Max ``||decode(encode(Z)) - Z||`` over ``Z_list``, relative to the
        largest tangent in the set.

        Scaled by the largest norm rather than each vector's own. With the usual
        ``reference="middle"`` the reference is itself a training point, so one
        tangent is exactly zero and a per-vector relative error divides roundoff
        by zero. Measuring the whole set against its own scale is what the
        question, does the chart represent these tangents, actually asks.
        """
        Z_list = list(Z_list)
        if not Z_list:
            return 0.0
        scale = max(float(norm(Z)) for Z in Z_list)
        if scale == 0.0:
            return 0.0
        worst = 0.0
        for Z in Z_list:
            d = float(norm(self.decode(self.encode(Z)) - Z)) / scale
            worst = max(worst, d)
        return worst


class ExactChart(Chart):
    """Isometric chart built from the horizontal basis ``F`` at ``Phi0``."""

    def __init__(self, Phi0: np.ndarray, F: np.ndarray | None = None):
        self.shape = Phi0.shape
        n, r = self.shape
        self.F = horizontal_basis(Phi0) if F is None else np.asarray(F, dtype=float)
        if self.F.shape[1] != n * r:
            raise ValueError(
                f"F has {self.F.shape[1]} columns but vec(Z) has {n * r} entries"
            )
        self.dim = self.F.shape[0]

    @classmethod
    def from_file(cls, Phi0: np.ndarray, path) -> "ExactChart":
        """Load a precomputed ``F_matrix.npy`` (the cylinder pipeline saves one)."""
        return cls(Phi0, F=np.load(path))

    def encode(self, Z: np.ndarray) -> np.ndarray:
        return self.F @ np.asarray(Z, dtype=float).flatten("F")

    def decode(self, y: np.ndarray) -> np.ndarray:
        return (self.F.T @ np.asarray(y, dtype=float)).reshape(self.shape, order="F")

    def tangent_norm(self, y: np.ndarray) -> float:
        # F has orthonormal rows, so ||F.T y|| == ||y||.
        return float(norm(y))

    def ball_params(self, radius: float) -> tuple[np.ndarray, float]:
        return np.zeros(self.dim), float(radius)


class PCAChart(Chart):
    """PCA over training tangent vectors; exact norms via a small correction.

    With ``center=True`` (default) the encoding subtracts the tangent mean, so
    ``||y|| != ||Z||_F``. :meth:`tangent_norm` corrects for that exactly using the
    cached projection of the mean, which keeps the ``pi/2`` constraint meaningful.
    Set ``center=False`` to make the chart a plain isometry at the cost of
    spending one component on the mean direction.
    """

    def __init__(self, Z_train, max_dim: int = 20, center: bool = True):
        Z_train = list(Z_train)
        if not Z_train:
            raise ValueError("PCAChart needs at least one training tangent vector")
        self.shape = Z_train[0].shape
        self.center = center

        M = np.vstack([np.asarray(Z, dtype=float).flatten("F") for Z in Z_train])
        self.mean = M.mean(axis=0) if center else np.zeros(M.shape[1])
        C = M - self.mean

        keep = max(1, min(len(Z_train) - (1 if center else 0), max_dim))
        keep = min(keep, min(C.shape))

        # For a large full-order dimension and comparatively few parameter
        # samples, diagonalise the sample-space Gram matrix instead of asking
        # LAPACK for the SVD of a very wide dense matrix in every CV fold.
        # C = U S V^T implies C C^T = U S^2 U^T and V = C^T U S^-1,
        # so this is algebraically the same PCA chart, not an approximation.
        if C.shape[1] > 2 * C.shape[0]:
            eigvals, U = np.linalg.eigh(C @ C.T)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            U = U[:, order]
            S_all = np.sqrt(eigvals)
            positive = S_all > np.finfo(float).eps * max(S_all[0], 1.0) * max(C.shape)
            keep = min(keep, int(np.sum(positive)))
            if keep < 1:
                raise ValueError("training tangent vectors have no nonzero PCA direction")
            S = S_all[:keep]
            self.V = (C.T @ U[:, :keep]) / S
            # Roundoff in the Gram route is removed with a tiny QR; this also
            # keeps the norm correction below exact to working precision.
            self.V, _ = np.linalg.qr(self.V, mode="reduced")
            total = float(np.sum(eigvals)) + 1e-30
        else:
            _, S_all, Vt = svd(C, full_matrices=False)
            S = S_all[:keep]
            self.V = Vt[:keep].T  # (nr, m)
            total = float(np.sum(S_all**2)) + 1e-30
        self.singular_values = S
        self.dim = keep

        self.explained_variance_ratio = np.cumsum(S**2) / total

        # Cached pieces for the exact norm of mean + V @ y.
        self._Vt_mean = self.V.T @ self.mean          # (m,)
        self._mean_sq = float(self.mean @ self.mean)  # scalar

    def encode(self, Z: np.ndarray) -> np.ndarray:
        return self.V.T @ (np.asarray(Z, dtype=float).flatten("F") - self.mean)

    def decode(self, y: np.ndarray) -> np.ndarray:
        return (self.V @ np.asarray(y, dtype=float) + self.mean).reshape(
            self.shape, order="F"
        )

    def tangent_norm(self, y: np.ndarray) -> float:
        # ||V y + mu||^2 = ||y||^2 + 2 y.(V^T mu) + ||mu||^2   (V orthonormal cols)
        y = np.asarray(y, dtype=float)
        sq = float(y @ y) + 2.0 * float(y @ self._Vt_mean) + self._mean_sq
        return float(np.sqrt(max(sq, 0.0)))

    def ball_params(self, radius: float) -> tuple[np.ndarray, float]:
        # ||V y + mu||^2 <= R^2  <=>  ||y + V^T mu||^2 <= R^2 - ||mu||^2 + ||V^T mu||^2
        b = self._Vt_mean
        rho_sq = radius**2 - self._mean_sq + float(b @ b)
        if rho_sq <= 0:
            raise ValueError(
                # the test is rho_sq <= 0, that is the component of the mean
                # orthogonal to the chart subspace already fills the ball.
                # ||mean|| >= radius is necessary for that but not sufficient,
                # so quoting it alone misdescribed the condition that fired.
                "the displaced ball is empty: the component of the training tangent "
                "mean orthogonal to the chart subspace already fills the injectivity "
                f"ball (||mean||={np.sqrt(self._mean_sq):.3f}, "
                f"||V^T mean||={np.sqrt(float(b @ b)):.3f}, radius={radius:.3f}); "
                "the reference basis is too far from the data"
            )
        return b, float(np.sqrt(rho_sq))


def build_chart(Phi0: np.ndarray, Z_train, kind: str = "auto", max_dim: int = 20,
                center: bool = True, exact_max_dim: int = 4000) -> Chart:
    """Pick a chart. ``"auto"`` uses ``ExactChart`` only when ``nr`` is small enough."""
    n, r = Phi0.shape
    if kind == "exact" or (kind == "auto" and n * r <= exact_max_dim):
        return ExactChart(Phi0)
    if kind in ("auto", "pca"):
        return PCAChart(Z_train, max_dim=max_dim, center=center)
    raise ValueError(f"unknown chart kind {kind!r}")
