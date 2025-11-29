#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --job-name=isorank-serial-4
#SBATCH --output=results/serial_4_%A_%a.out
#SBATCH --error=results/serial_4_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# Correct Python environment
module load python/anaconda-2020.02

# Enable OpenMP / MKL multi-thread
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p results

GRAPHS=(
    benchmarks/bench_S2k.npz
    benchmarks/bench_M5k.npz
    benchmarks/bench_L10k.npz
)

GRAPH="${GRAPHS[$SLURM_ARRAY_TASK_ID]}"

echo "Running SERIAL OpenMP-Accelerated IsoRank on $GRAPH ..."
python isorank_serial.py \
    --graph "$GRAPH" \
    --alpha 0.9 \
    --max_iter 30 \
    --tol 1e-4 \
    --quiet

echo "Finished SERIAL OpenMP-Accelerated IsoRank on $(hostname) at $(date)"
