#!/bin/bash
#SBATCH -J Analysis
### number of cores
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p zencloud
#SBATCH --mem=20G
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

module load python/3.12
source ../.venv/bin/activate
####### SETTING ENV VARIABLE #######
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
####################################

####### RUN #######
python transfer_operator.py