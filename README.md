# Distributed IsoRank (MPI + Python)

This project implements a distributed IsoRank algorithm using MPI row-block parallelism.
Synthetic scale-free graph benchmarks are used to evaluate strong scaling performance.

---

## Attributions

This work was completed as part of **UChicago MPCS 56430** course.

References:
- mpi4py, NumPy, SciPy, NetworkX
- Singh et al., *Global alignment of multiple protein interaction networks*, 2008

---

## Benchmark Files

Already included in `benchmarks/`.

Optional regeneration:
python3 make_benchmarks.py --outdir benchmarks --sets S2k M5k L10k

Graph sizes:
- **S2k**: 2,000 nodes
- **M5k**: 5,000 nodes
- **L10k**: 10,000 nodes  
(Scale-free / BA structure)

---

##  Run Serial IsoRank
sbatch run_serial.sbatch

Output:
R_results/serial_R.npy

---

##  Run Distributed MPI IsoRank

Example: 4 ranks
sbatch run_mpi_4.sbatch

Output:
R_results/mpi_R_4ranks.npy

---

##  Validation

Compare serial vs MPI computation:

python3 validation.py
--serial R_results/serial_R.npy
--mpi R_results/mpi_R_4ranks.npy

Expected outputs:
- **L1/L2 diff → near zero**
- **Cosine similarity > 0.9999**
- **Top-1 match rate ≈ 1.0**
- **Sum(R)=1** and **min(R) ≥ 0**

This confirms the distributed result matches serial correctness.

---

## Notes

- Strong scaling is limited by Allreduce/Allgather synchronization overhead
- Best improvements observed for large graphs due to higher compute/communication ratio

---

## Author

**Oz Zhou**  
University of Chicago · MPCS 56430 Project

---