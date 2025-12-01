#!/bin/bash
#SBATCH -A mpcs56430
#SBATCH --partition=bigmem2
#SBATCH --job-name=iso-mpi-8
#SBATCH --output=results/mpi_8_%A_%a.out
#SBATCH --error=results/mpi_8_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=480G

module load intelmpi/2018.4.274+intel-18.0.5
module load mpi4py/3.0.3+intelmpi-2018.4.274

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

echo "[MPI-8] Running IsoRank on ${GRAPH}..."

mpirun -n 8 python isorank_mpi.py \
    --graph "${GRAPH}" \
    --alpha 0.9 \
    --max_iter 30 \
    --tol 1e-4 \
    --quiet

echo "[MPI-8] Finished ${GRAPH} on $(hostname) at $(date)"
