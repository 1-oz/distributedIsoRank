#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time
import numpy as np
from scipy.sparse import csr_matrix
from utils import load_graph_pair


def build_P(adj: csr_matrix):
    """Column-normalized sparse matrix"""
    degrees = np.array(adj.sum(axis=0)).flatten()
    degrees[degrees == 0] = 1.0
    inv_deg = 1.0 / degrees
    n = len(inv_deg)
    D_inv = csr_matrix((inv_deg, (np.arange(n), np.arange(n))), shape=(n, n))
    return adj.dot(D_inv)


def build_synthetic_E(n1, n2, seed=42, diag_strength=1.0):
    """Diagonal-dominant, dense synthetic similarity"""
    np.random.seed(seed)
    E = np.random.rand(n1, n2) * 0.1
    for i in range(min(n1, n2)):
        E[i, i] += diag_strength
    E /= E.sum()
    return E


def isorank(P, Q_dense, E, alpha, max_iter=20, tol=1e-4, verbose=True):
    """
    IsoRank Iteration:
        R = alpha * P R Q^T + (1-alpha) * E

    Key guarantee:
        - R always dense np.ndarray
        - Q always dense
    """
    R = E.copy()
    start = time.time()

    for k in range(max_iter):

        # sparse (P) x dense (R) --> dense
        R_new = P.dot(R)
        R_new = np.array(R_new)

        # dense x dense (Q^T) --> dense
        R_new = R_new.dot(Q_dense.T)

        # IsoRank update (fully dense now)
        R_new = alpha * R_new + (1 - alpha) * E
        R_new /= R_new.sum()

        diff = np.linalg.norm(R_new - R, 1)
        if verbose:
            print(f"Iter {k:02d}: diff={diff:.4e}")
        if diff < tol:
            break

        R = R_new

    elapsed = time.time() - start
    return R, elapsed, k+1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diag", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"\nLoading graphs from {args.graph} ...")
    A1, _, A2, _ = load_graph_pair(args.graph)
    n1, n2 = A1.shape[0], A2.shape[0]
    print(f"Graph sizes: G1={n1}, G2={n2}")

    print("Constructing P, Q ...")
    P = build_P(A1)
    Q = build_P(A2)
    Q_dense = Q.toarray()      # <<<<< FIX CORE HERE

    print("Initializing E ...")
    E = build_synthetic_E(n1, n2, seed=args.seed, diag_strength=args.diag)

    print("\nRunning Serial IsoRank ...")
    R, elapsed, iters = isorank(
        P, Q_dense, E,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=not args.quiet
    )

    print("\n=== DONE ===")
    print(f"Graph: {args.graph}")
    print(f"alpha={args.alpha}")
    print(f"Runtime: {elapsed:.2f} s")
    print(f"Iterations: {iters}\n")


if __name__ == "__main__":
    main()
