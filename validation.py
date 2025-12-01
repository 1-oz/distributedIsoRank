#!/usr/bin/env python3
import numpy as np
import argparse

def validate(serial_path, mpi_path):
    print(f"\n[Validation] Serial: {serial_path}")
    print(f"[Validation] MPI:    {mpi_path}\n")

    R_s = np.load(serial_path)
    R_m = np.load(mpi_path)

    # shape check
    if R_s.shape != R_m.shape:
        print("ERROR: Shape mismatch!")
        print(f"Serial: {R_s.shape}, MPI: {R_m.shape}")
        return

    # norm differences
    diff_l1 = np.linalg.norm(R_s - R_m, 1)
    diff_l2 = np.linalg.norm(R_s - R_m)
    cosine = np.sum(R_s * R_m) / (np.linalg.norm(R_s) * np.linalg.norm(R_m))

    # alignment check (Top-1 argmax) 
    top1_s = np.argmax(R_s, axis=1)
    top1_m = np.argmax(R_m, axis=1)
    top1_match = np.mean(top1_s == top1_m)

    # probability checks
    sum_check = np.abs(R_m.sum() - 1.0)
    min_check = R_m.min()

    print("Validation Results")
    print(f"L1 diff:             {diff_l1:.6e}")
    print(f"L2 diff:             {diff_l2:.6e}")
    print(f"Cosine similarity:   {cosine:.6f}")
    print(f"Top-1 match rate:    {top1_match:.4f}")
    print(f"Sum(R)=1 check:      abs(sum-1)={sum_check:.2e}")
    print(f"Min(R)>=0 check:     min(R)={min_check:.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, required=True)
    parser.add_argument("--mpi", type=str, required=True)
    args = parser.parse_args()

    validate(args.serial, args.mpi)
