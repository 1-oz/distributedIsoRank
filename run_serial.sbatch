#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --partition=bigmem2
#SBATCH --job-name=isorank-serial-4
#SBATCH --output=results/serial_4_%A_%a.out
#SBATCH --error=results/serial_4_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=128G

module load python/3.9.18

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

RESULT_DIR="results/${EXPTYPE}"
mkdir -p "$RESULT_DIR"

if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "S" ]; then
    GRAPHS=(
        benchmarks/bench_S2k.npz
        benchmarks/bench_M5k.npz
        benchmarks/bench_L10k.npz
    )
elif [ "$EXPTYPE" == "L" ]; then
    GRAPHS=(
        benchmarks/bench_L.npz
        benchmarks/bench_XL.npz
        benchmarks/bench_XXL.npz
    )
else
    echo "Invalid EXPTYPE value! Use S or L."
    exit 1
fi

GRAPH=${GRAPHS[$SLURM_ARRAY_TASK_ID]}

echo "Running SERIAL OpenMP-Accelerated IsoRank on $GRAPH ..."
python3 isorank_serial.py \
    --graph "$GRAPH" \
    --alpha 0.9 \
    --max_iter 30 \
    --tol 1e-4 \
    --quiet

echo "Finished SERIAL OpenMP-Accelerated IsoRank on $(hostname) at $(date)"
