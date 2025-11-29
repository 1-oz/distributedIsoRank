#!/usr/bin/env python3
"""
Distributed IsoRank using MPI (row-block parallelism)
Author: Oz Zhou

Run with e.g.:
    srun -A mpcs56430 -p short -n 4 --time=00:10:00 \
        python isorank_mpi.py \
            --graph benchmarks/bench_S2k.npz \
            --alpha 0.9 --max_iter 20 --tol 1e-4 --quiet
"""
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

    P_block = P[row_start:row_end, :]     # (n_local x n1) 稀疏
    E_block = E[row_start:row_end, :]     # (n_local x n2) 稠密

    R = E.copy()
    R_block = R[row_start:row_end, :]

    if rank == 0 and verbose:
        print(f"[IsoRank-MPI] n1={n1}, n2={n2}, world_size={world_size}")
        print(f"[IsoRank-MPI] rank0 rows [{row_start}:{row_end})")

    start_time = time.time()

    for k in range(max_iter):
        # ----------------------------------------------------
        # Step 1: 所有 rank 计算 B = R Q^T （dense x dense）
        # ----------------------------------------------------
        # B: (n1 x n2)
        B = R.dot(Q_dense.T)

        # ----------------------------------------------------
        # Step 2: 每个 rank 只用自己的 P_block 计算局部块：
        #         R_block_new = P_block * B
        # ----------------------------------------------------
        R_block_new = P_block.dot(B)          # (n_local x n2)
        R_block_new = np.array(R_block_new)   # 确保是 ndarray

        # ----------------------------------------------------
        # Step 3: 本地块混入 E_block，得到：
        #         R_block_new = alpha * R_block_new + (1-alpha) * E_block
        # ----------------------------------------------------
        R_block_new = alpha * R_block_new + (1.0 - alpha) * E_block

        # 归一化：需要整个 R_new 的全局和
        local_sum = R_block_new.sum()
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)
        R_block_new /= global_sum

        # ----------------------------------------------------
        # Step 4: 计算全局 diff = ||R_new - R||_1
        #         每个 rank 算自己块的 diff，再 allreduce
        # ----------------------------------------------------
        local_diff = np.linalg.norm(R_block_new - R_block, 1)
        global_diff = comm.allreduce(local_diff, op=MPI.SUM)

        if rank == 0 and verbose:
            print(f"Iter {k:02d}: diff={global_diff:.4e}")

        # ----------------------------------------------------
        # Step 5: allgather 所有 rank 的 row-block，拼成完整 R_new
        #         这样下一轮每个 rank 又有完整的 R
        # ----------------------------------------------------
        blocks = comm.allgather(R_block_new)
        R_new_full = np.vstack(blocks)

        # 收敛判断只在 rank 0 做，然后广播给所有 rank
        converged = (global_diff < tol)
        converged = comm.bcast(converged, root=0)

        # 更新 R / R_block
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

    # 只在 rank 0 解析命令行参数，再广播给所有 rank
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

    # ----- 所有 rank 都加载图（为简单起见；规模在 10k 级别没问题） -----
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

    # ----- 运行分布式 IsoRank -----
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

    # 只在 rank 0 打印结果
    if rank == 0:
        print("\n=== MPI IsoRank DONE ===")
        print(f"Graph: {args.graph}")
        print(f"alpha={args.alpha}")
        print(f"MPI world size: {world_size}")
        print(f"Runtime: {elapsed:.2f} s")
        print(f"Iterations: {iters}\n")


if __name__ == "__main__":
    main()
