#!/usr/bin/env python3
import argparse
import time
import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix
from utils import load_graph_pair

def build_P(adj):
    degrees = np.array(adj.sum(axis=0)).flatten()
    degrees[degrees == 0] = 1.0
    inv_deg = 1.0 / degrees
    n = len(inv_deg)
    D_inv = csr_matrix((inv_deg, (np.arange(n), np.arange(n))), shape=(n, n))
    return adj.dot(D_inv)


def build_synthetic_E(n1, n2, seed=42, diag_strength=1.0):
    np.random.seed(seed)
    E = np.random.rand(n1, n2) * 0.1
    for i in range(min(n1, n2)):
        E[i, i] += diag_strength
    E /= E.sum()
    return E



def compute_row_range(n_rows, world_size, rank):
    base = n_rows // world_size
    rem = n_rows % world_size
    if rank < rem:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end



def isorank_mpi(P, Q_dense, E, alpha, comm,
                max_iter=20, tol=1e-4, verbose=True):
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    n1, n2 = E.shape

    row_start, row_end = compute_row_range(n1, world_size, rank)
    n_local = row_end - row_start

    P_block = P[row_start:row_end, :]
    E_block = E[row_start:row_end, :]

    R = E.copy()
    R_block = R[row_start:row_end, :]

    if rank == 0 and verbose:
        print(f"[IsoRank-MPI] n1={n1}, n2={n2}, world_size={world_size}")
        print(f"[IsoRank-MPI] rank0 rows [{row_start}:{row_end})")

    start_time = time.time()

    for k in range(max_iter):
        B = R.dot(Q_dense.T)

        R_block_new = P_block.dot(B) 
        R_block_new = np.array(R_block_new)


        R_block_new = alpha * R_block_new + (1.0 - alpha) * E_block
        local_sum = R_block_new.sum()
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)
        R_block_new /= global_sum


        local_diff = np.linalg.norm(R_block_new - R_block, 1)
        global_diff = comm.allreduce(local_diff, op=MPI.SUM)

        if rank == 0 and verbose:
            print(f"Iter {k:02d}: diff={global_diff:.4e}")


        blocks = comm.allgather(R_block_new)
        R_new_full = np.vstack(blocks)

        converged = (global_diff < tol)
        converged = comm.bcast(converged, root=0)

        R = R_new_full
        R_block = R[row_start:row_end, :]

        if converged:
            if rank == 0 and verbose:
                print(f"[IsoRank-MPI] Converged at iter {k+1}")
            break

    elapsed = time.time() - start_time
    iters = k + 1
    return R, elapsed, iters


# ========= CLI & main =========

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--graph", type=str, required=True)
        parser.add_argument("--alpha", type=float, default=0.9)
        parser.add_argument("--max_iter", type=int, default=20)
        parser.add_argument("--tol", type=float, default=1e-4)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--diag", type=float, default=1.0)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()
    else:
        args = None

    args = comm.bcast(args, root=0)

    if rank == 0 and not args.quiet:
        print(f"\n[IsoRank-MPI] Loading graphs from {args.graph} ...")
    A1, _, A2, _ = load_graph_pair(args.graph)
    n1, n2 = A1.shape[0], A2.shape[0]

    if rank == 0 and not args.quiet:
        print(f"Graph sizes: G1={n1}, G2={n2}")
        print(f"MPI world size: {world_size}\n")
        print("Building P, Q ...")

    P = build_P(A1)
    Q = build_P(A2)
    Q_dense = Q.toarray()

    if rank == 0 and not args.quiet:
        print("Initializing E ...")

    E = build_synthetic_E(n1, n2, seed=args.seed, diag_strength=args.diag)
    if rank == 0 and not args.quiet:
        print("\nRunning IsoRank (MPI) ...")

    R, elapsed, iters = isorank_mpi(
        P, Q_dense, E,
        alpha=args.alpha,
        comm=comm,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=(not args.quiet)
    )

    if rank == 0:
        print("\n MPI IsoRank DONE")
        print(f"Graph: {args.graph}")
        print(f"alpha={args.alpha}")
        print(f"MPI world size: {world_size}")
        print(f"Runtime: {elapsed:.2f} s")
        print(f"Iterations: {iters}\n")
    
    save_path = f"R_results/mpi_R_{world_size}ranks.npy"
    np.save(save_path, R)
    print(f"[Saved] MPI result --> {save_path}")


if __name__ == "__main__":
    main()
