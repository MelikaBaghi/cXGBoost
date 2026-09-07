"""Grassmann manifold primitives: alignment, log/exp maps, error metrics.

A POD basis is only defined up to rotation of its columns, so the object we
actually regress is the subspace it spans -- a point on the Grassmannian
G(r, n). Everything here works on ``(n, r)`` matrices with orthonormal columns.

The exponential map at a reference point Phi0 is injective on the open ball of
radius pi/2 in the tangent space (measured by the largest principal angle). That
ball is what the constrained boosting in :mod:`cxgboost.boosting` stays inside.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, norm, qr, svd

#: Injectivity radius of the Grassmann exponential map. The map is one-to-one
#: where the largest principal angle is *strictly* below this, so ``PI_2``
#: itself is the cut locus and not a feasible target.
PI_2 = np.pi / 2

#: Margin that pulls the enforced ball strictly inside the one-to-one region.
#: The closed ball of radius ``PI_2`` touches the cut locus: a rank-one tangent
#: with ``||Z||_F = PI_2`` also has ``||Z||_2 = PI_2``, so it sits exactly where
#: the map stops being injective. Enforcing ``RHO`` instead makes the closed
#: feasible set strictly interior, which is what lets the guarantee be stated
#: without a separate repair step afterwards.
EPS_INJ = 1e-9

#: The radius actually enforced everywhere in this package.
RHO = PI_2 - EPS_INJ

LOG_VARIANTS = ("projected", "affine", "legacy")


def orthonormalize(Phi: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column span of ``Phi``.

    Returns the left singular vectors, so the *span* is preserved but the
    individual columns are not: for an already-orthonormal ``Phi`` every singular
    value is 1, the SVD is non-unique, and this returns ``Phi V`` for an
    arbitrary rotation ``V``. That is fine wherever only the subspace matters
    (which is everywhere on the Grassmannian) but **wrong** if the columns are
    paired with stored POD coefficients -- use :func:`nearest_orthonormal` there.
    """
    U, _, _ = svd(np.asarray(Phi, dtype=float), full_matrices=False)
    return U


def nearest_orthonormal(Phi: np.ndarray) -> np.ndarray:
    """Closest orthonormal matrix to ``Phi`` in Frobenius norm (polar factor).

    ``U V^T`` from ``Phi = U S V^T``. Unlike :func:`orthonormalize` this is a
    genuine minimal correction — it reduces to the identity map when ``Phi`` is
    already orthonormal — so column identity, and therefore the pairing with a
    stored ``sigmas``/``Vt``, survives. Use it to clean up float32 round-off.
    """
    U, _, Vt = svd(np.asarray(Phi, dtype=float), full_matrices=False)
    return U @ Vt


def align(Phi: np.ndarray, Phi0: np.ndarray) -> np.ndarray:
    """Rotate ``Phi``'s columns to the representative closest to ``Phi0``.

    Orthogonal Procrustes: ``argmin_R ||Phi R - Phi0||_F`` over ``R`` in O(r).
    Both matrices span the same subspaces before and after; this only removes
    the rotational ambiguity so that differences are meaningful.
    """
    U, _, Vt = svd(Phi.T @ Phi0, full_matrices=False)
    return Phi @ (U @ Vt)


def log_map(
    Phi: np.ndarray,
    Phi0: np.ndarray,
    variant: str = "projected",
    tol: float = 1e-12,
) -> np.ndarray:
    """Tangent vector ``Z`` at ``Phi0`` pointing at the subspace of ``Phi``.

    ``Z`` is horizontal (``Phi0.T @ Z == 0``) and its singular values are the
    principal angles between the two subspaces, so ``||Z||_2`` is the largest
    principal angle and the injectivity condition reads ``||Z||_2 < pi/2``.

    variant
        ``"projected"``  -- ``Z = U arcsin(S) V.T`` with ``U S V.T = (I - Phi0 Phi0.T) Phi``.
                            The singular values of the projected difference are
                            ``sin(theta_k)``, so ``arcsin`` recovers the angles.
        ``"affine"``     -- textbook form, ``U arctan(S) V.T`` with
                            ``U S V.T = Phi (Phi0.T Phi)^-1 - Phi0`` (singular values
                            are ``tan(theta_k)``). Mathematically identical to
                            ``"projected"``; slightly worse conditioned near pi/2.
        ``"legacy"``     -- ``arctan`` applied to the *projected* singular values.
                            This is what the original notebooks used. It is **not**
                            the inverse of :func:`exp_map` (it under-estimates the
                            angles), and is kept only to reproduce published numbers.

    See ``docs/reproducibility.md`` for which variant each published table used.
    """
    if variant not in LOG_VARIANTS:
        raise ValueError(f"variant must be one of {LOG_VARIANTS}, got {variant!r}")

    n = Phi0.shape[0]
    Phi = align(orthonormalize(Phi), Phi0)

    if variant == "affine":
        A = Phi @ inv(Phi0.T @ Phi) - Phi0
        if norm(A, "fro") < tol:
            return np.zeros_like(Phi0)
        U, S, Vt = svd(A, full_matrices=False)
        angles = np.arctan(S)
    else:
        M = Phi - Phi0 @ (Phi0.T @ Phi)  # (I - Phi0 Phi0.T) Phi
        if norm(M, "fro") < tol:
            return np.zeros_like(Phi0)
        U, S, Vt = svd(M, full_matrices=False)
        angles = np.arctan(S) if variant == "legacy" else np.arcsin(np.clip(S, -1.0, 1.0))

    Z = U @ np.diag(angles) @ Vt
    return Z - Phi0 @ (Phi0.T @ Z)  # re-project onto the horizontal space


def exp_map(Z: np.ndarray, Phi0: np.ndarray) -> np.ndarray:
    """Move from ``Phi0`` along the geodesic with initial velocity ``Z``.

    Returns an orthonormal basis of the resulting subspace. Injective for
    ``||Z||_2 < pi/2``; beyond that different ``Z`` can map to the same subspace
    and the inverse relation with :func:`log_map` breaks down.
    """
    U, S, Vt = svd(Z, full_matrices=False)
    Phi = (Phi0 @ Vt.T @ np.diag(np.cos(S)) + U @ np.diag(np.sin(S))) @ Vt
    return orthonormalize(Phi)


def clip_to_ball(Z: np.ndarray, radius: float = PI_2 - 1e-6, ord="fro") -> np.ndarray:
    """Radially rescale ``Z`` back inside the ball of the given ``radius``.

    The test-time safeguard of Proposition 2. Two notions of "inside" are in play
    and they are not the same:

    ``ord="fro"`` (default)
        ``||Z||_F <= radius``. This is the condition the training constraint
        enforces, so the safeguard matches the method. It is *conservative*:
        since ``||Z||_2 <= ||Z||_F``, it can rescale tangent vectors that were
        already injective, and it bites harder the larger ``r`` is.
    ``ord=2``
        ``||Z||_2 <= radius``, the exact injectivity condition (the largest
        principal angle). Use this to measure how much the Frobenius bound
        over-constrains a given benchmark.
    """
    s = norm(Z, ord)
    return Z * (radius / s) if s > radius else Z


def principal_angles(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two column spans.

    ``svd`` returns the cosines in descending order and ``arccos`` is decreasing,
    so the angles come out ascending already -- no reversal.
    """
    s = svd(orthonormalize(P).T @ orthonormalize(Q), compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def geodesic_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Grassmann geodesic distance, i.e. the 2-norm of the principal angles."""
    return float(norm(principal_angles(P, Q)))


def projection_error(Phi: np.ndarray, D: np.ndarray) -> float:
    """Relative field reconstruction error ``||D - Phi Phi.T D||_F / ||D||_F``.

    This is the metric reported throughout the paper: how well the *predicted*
    subspace represents the *true* snapshots. Independent of the basis rotation,
    which is why it is the right quantity for a subspace-valued prediction.
    """
    Phi = orthonormalize(Phi)
    den = norm(D, "fro")
    if den == 0:
        return 0.0
    return float(norm(D - Phi @ (Phi.T @ D), "fro") / den)


def snapshots_from_svd(Phi: np.ndarray, sigmas: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Rebuild the rank-r snapshot matrix ``D = Phi diag(sigmas) Vt``."""
    return Phi @ np.diag(sigmas) @ Vt


def horizontal_basis(Phi0: np.ndarray) -> np.ndarray:
    """Orthonormal rows of an isometric coordinate system containing the
    horizontal space at ``Phi0``, returned as ``F``.

    Careful with the dimension. The horizontal space at ``Phi0`` is
    ``{Z : Phi0.T @ Z == 0}``, which is ``r**2`` scalar constraints and therefore
    has dimension ``(n - r) r = nr - r**2``. The rows built here impose only the
    ``r`` diagonal conditions ``Phi0[:, i] . Z[:, i] == 0``, so ``F`` has
    ``(n - 1) r = nr - r`` rows and spans a *larger* space that contains the
    horizontal space as a proper subspace. (Reshaped rows of ``F`` reach
    ``||Phi0.T @ Z||_F ~ 0.7``, so they are not themselves horizontal.)

    This is harmless and deliberate: ``F F.T = I``, so on genuine tangent
    vectors, which is all this is ever applied to, the map is an isometry and
    ``||F vec(Z)|| == ||Z||_F`` exactly. It is an over-parametrised but exact
    coordinate system, not a basis for the horizontal space. Do not quote
    ``nr - r`` as the dimension of the horizontal space.

    Costs ``O(n^2 r)`` memory -- only usable for small ``n`` (the cylinder case).
    Larger benchmarks use :class:`cxgboost.chart.PCAChart` instead.
    """
    n, r = Phi0.shape
    rows = []
    for i in range(r):
        Q, _ = qr(Phi0[:, i].reshape(-1, 1), mode="complete")
        comp = Q[:, 1:]  # orthogonal complement of column i, (n, n-1)
        for j in range(n - 1):
            row = np.zeros(n * r)
            row[i * n : (i + 1) * n] = comp[:, j]
            rows.append(row)
    return np.vstack(rows)
