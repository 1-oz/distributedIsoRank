#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --job-name=iso-mpi-16
#SBATCH --output=results/mpi_16_%A_%a.out
#SBATCH --error=results/mpi_16_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=64G

module load intelmpi/2018.4.274+intel-18.0.5
module load mpi4py/3.0.3+intelmpi-2018.4.274

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p results

GRAPHS=(
    benchmarks/bench_S2k.npz
    benchmarks/bench_M5k.npz
    benchmarks/bench_L10k.npz
)

GRAPH=${GRAPHS[$SLURM_ARRAY_TASK_ID]}

echo "[MPI-16] Running IsoRank on ${GRAPH}..."

mpirun -n 16 python isorank_mpi.py \
    --graph "${GRAPH}" \
    --alpha 0.9 \
    --max_iter 30 \
    --tol 1e-4 \
    --quiet

echo "[MPI-16] Finished ${GRAPH} on $(hostname) at $(date)"
