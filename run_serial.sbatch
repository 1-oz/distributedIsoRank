#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --job-name=isorank-serial
#SBATCH --output=results/serial_%A_%a.out
#SBATCH --error=results/serial_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

module load python/3.9.18

mkdir -p results

GRAPHS=(
    benchmarks/bench_T500.npz
    benchmarks/bench_S2k.npz
    benchmarks/bench_M5k.npz
)

GRAPH="${GRAPHS[$SLURM_ARRAY_TASK_ID]}"

echo "Running serial IsoRank on $GRAPH ..."
python3 isorank_serial.py \
    --graph "$GRAPH" \
    --alpha 0.9 \
    --max_iter 30 \
    --tol 1e-4 \
    --quiet

echo "Completed $GRAPH on $(hostname) at $(date)"
