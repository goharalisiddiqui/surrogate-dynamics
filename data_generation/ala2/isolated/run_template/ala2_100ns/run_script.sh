#!/bin/bash
#SBATCH -J MetaD
### number of cores
#SBATCH -c 4
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -p zencloud
#SBATCH -B 1:16:2
#SBATCH --mem=20G
#SBATCH --gpus=0
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o /home/ge45daw/slurm_logs/slurm-%J.out
#SBATCH -e /home/ge45daw/slurm_logs/slurm-%J.err

####### PREPARE ENV #######
echo "Loading modules from $SLURM_PRESCRIPT_GROMACS :"
cat $SLURM_PRESCRIPT_GROMACS
source $SLURM_PRESCRIPT_GROMACS
########################################

####### SETTING ENV VARIABLE #######
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
####################################

####### RUN #######
cp ../initialization/md.tpr .
srun gmx_mpi mdrun -deffnm md &> run.log