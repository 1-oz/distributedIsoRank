#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --job-name=iso-mpi-w2
#SBATCH --output=results/mpi_w2_%A_%a.out
#SBATCH --error=results/mpi_w2_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

module load python/3.9.18
module load openmpi

mkdir -p results_mpi

GRAPHS=(
    benchmarks/bench_S2k.npz
    benchmarks/bench_M5k.npz
    benchmarks/bench_L10k.npz
)

GRAPH=${GRAPHS[$SLURM_ARRAY_TASK_ID]}

echo "[MPI-W2] Running IsoRank on $GRAPH..."

srun python3 isorank_mpi.py \
    --graph "$GRAPH" \
    --alpha 0.9 \
    --max_iter 30 \
    --tol 1e-4 \
    --quiet

echo "[MPI-W2] Completed $GRAPH on $(hostname) at $(date)"
