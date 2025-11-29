#!/usr/bin/env python3

import argparse
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

def generate_graph(n_nodes=1000, avg_degree=6, model="er", seed=42):
    np.random.seed(seed)

    if model == "er":
        # Erdos-Renyi random graph
        p = avg_degree / (n_nodes - 1)
        G = nx.fast_gnp_random_graph(n_nodes, p, seed=seed)

    elif model == "ba":
        # Barabási–Albert power-law graph
        m = avg_degree // 2 if avg_degree >= 2 else 1
        G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)

    else:
        raise ValueError("Unknown model type: choose 'er' or 'ba'")

    # Remove isolated nodes to avoid issues in IsoRank normalization
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        G = nx.convert_node_labels_to_integers(G)

    return G


def to_sparse_adj_matrix(G):
    """ Returns adjacency matrix (CSR sparse) and degree array """
    A = nx.to_scipy_sparse_array(G, format="csr")
    degrees = np.array(A.sum(axis=0)).flatten()
    return A, degrees


def main():
    parser = argparse.ArgumentParser(description="Generate two synthetic graphs")
    parser.add_argument("--n1", type=int, default=2000)
    parser.add_argument("--n2", type=int, default=2000)
    parser.add_argument("--deg", type=int, default=6)
    parser.add_argument("--model", type=str, default="er", choices=["er", "ba"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print(f"Generating Graph1: n={args.n1}, deg={args.deg}, model={args.model}")
    G1 = generate_graph(args.n1, args.deg, args.model, args.seed)
    A1, d1 = to_sparse_adj_matrix(G1)

    print(f"Generating Graph2: n={args.n2}, deg={args.deg}, model={args.model}")
    G2 = generate_graph(args.n2, args.deg, args.model, args.seed + 1)
    A2, d2 = to_sparse_adj_matrix(G2)

    if args.save:
        # You can change paths to your project repo structure
        np.savez("synthetic_graphs.npz",
                 A1=A1.data, A1_indices=A1.indices, A1_indptr=A1.indptr,
                 d1=d1, n1=A1.shape[0],
                 A2=A2.data, A2_indices=A2.indices, A2_indptr=A2.indptr,
                 d2=d2, n2=A2.shape[0])
        print("Saved to synthetic_graphs.npz")

    print("Done.")


if __name__ == "__main__":
    main()
