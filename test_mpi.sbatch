#!/bin/bash
#SBATCH --job-name=testmpi
#SBATCH --output=testmpi_%j.out
#SBATCH --time=00:01:00
#SBATCH -A mpcs56430
#SBATCH --ntasks=2

module load intelmpi/2018.4.274+intel-18.0.5
module load mpi4py/3.0.3+intelmpi-2018.4.274

mpirun python3 test_mpi.py

