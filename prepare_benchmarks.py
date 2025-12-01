#!/usr/bin/env python3
import argparse
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from make_synthetic_graph import generate_graph, to_sparse_adj_matrix


def save_graph_pair(prefix, A1, A2, d1, d2):
    """
    Save two graphs in CSR format into a single NPZ file
    following a consistent naming scheme.
    """
    filename = f"{prefix}.npz"
    np.savez(filename,
             # Graph 1
             A1_data=A1.data, A1_indices=A1.indices, A1_indptr=A1.indptr, n1=A1.shape[0],
             d1=d1,
             # Graph 2
             A2_data=A2.data, A2_indices=A2.indices, A2_indptr=A2.indptr, n2=A2.shape[0],
             d2=d2)
    print(f"[SAVED] {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--sets", type=str, nargs="+",
                        default=["L", "XL", "XXL"],
                        help="Benchmark names to generate")
    args = parser.parse_args()

    benchmark_configs = {
        "T500": dict(n=500, deg=4, seeds=(11, 12)),
        "S2k": dict(n=2000, deg=6, seeds=(1, 2)),
        "M5k": dict(n=5000, deg=6, seeds=(3, 4)),
        "L10k": dict(n=10000, deg=6, seeds=(5, 6)),
        #"L": dict(n=10000, deg=10, seeds=(5, 6)),
        #"XL": dict(n=20000, deg=10, seeds=(7, 8)),
        #"XXL": dict(n=50000, deg=10, seeds=(9, 10)),
    }

    print("\nPreparing benchmark graph pairs...\n")
    for name in args.sets:
        conf = benchmark_configs[name]
        n = conf["n"]
        deg = conf["deg"]
        seed1, seed2 = conf["seeds"]

        print(f"Generating benchmark: {name}")
        print(f" - Nodes: {n}")
        print(f" - Avg degree: {deg}")
        print(f" - Seeds: {seed1} & {seed2}")

        G1 = generate_graph(n, deg, model="ba", seed=seed1)
        G2 = generate_graph(n, deg, model="ba", seed=seed2)

        A1, d1 = to_sparse_adj_matrix(G1)
        A2, d2 = to_sparse_adj_matrix(G2)

        prefix = f"{args.outdir}/bench_{name}"
        save_graph_pair(prefix, A1, A2, d1, d2)

    print("\n Benchmark graph suite generated.")


if __name__ == "__main__":
    main()
