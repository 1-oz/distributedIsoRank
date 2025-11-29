#!/usr/bin/env python3

import argparse
import time
import numpy as np
from scipy.sparse import csr_matrix
from utils import load_graph_pair


def build_P(adj: csr_matrix):
    """Column-normalized adjacency matrix"""
    degrees = np.array(adj.sum(axis=0)).flatten()
    degrees[degrees == 0] = 1
    inv_deg = 1.0 / degrees
    D_inv = csr_matrix((inv_deg, (range(len(inv_deg)), range(len(inv_deg)))))
    return adj.dot(D_inv)


def build_synthetic_E(n1, n2, seed=42, diag_strength=1.0):
    """Diagonal-dominant sequence similarity matrix"""
    np.random.seed(seed)
    E = np.random.rand(n1, n2) * 0.1
    for i in range(min(n1, n2)):
        E[i, i] += diag_strength
    E /= E.sum()
    return E


def isorank(P, Q, E, alpha, max_iter=50, tol=1e-6, verbose=True):
    """Iterative IsoRank solver"""
    R = E.copy()
    start = time.time()

    for k in range(max_iter):
        R_new = P.dot(R).dot(Q.T)       # core sparse-dense-dense step
        R_new = alpha * R_new + (1 - alpha) * E
        R_new /= R_new.sum()            # stochastic normalization

        diff = np.linalg.norm(R_new - R, 1)
        if verbose:
            print(f"Iter {k:02d}: diff={diff:.4e}")

        if diff < tol:
            break
        R = R_new
    
    elapsed = time.time() - start
    if verbose:
        print(f"Converged in {k+1} iters. Time={elapsed:.2f}s")

    return R, elapsed, k+1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, required=True,
                        help="Path to benchmark npz file")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--diag", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"\nLoading graphs from {args.graph} ...")
    A1, _, A2, _ = load_graph_pair(args.graph)
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    print(f"Graph sizes: G1={n1}, G2={n2}")

    print("Constructing P, Q ...")
    P = build_P(A1)
    Q = build_P(A2)

    print("Initializing E ...")
    E = build_synthetic_E(n1, n2, seed=args.seed, diag_strength=args.diag)

    print("\nRunning Serial IsoRank ...")
    verbose = not args.quiet
    _, elapsed, iters = isorank(
        P, Q, E,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=verbose
    )

    print("\n=== DONE ===")
    print(f"Graph: {args.graph}")
    print(f"alpha={args.alpha}")
    print(f"Runtime: {elapsed:.2f} s")
    print(f"Iterations: {iters}\n")


if __name__ == "__main__":
    main()
