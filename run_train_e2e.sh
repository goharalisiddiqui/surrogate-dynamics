#! /bin/bash
#SBATCH -J TFT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpucloud
#SBATCH --mem=8G
#SBATCH --gres=shard:1
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

unset $pref
if [ ! -z "${SLURM_JOB_ID}" ]; then
    echo "Running on compute node"
    pref='srun'
    mkdir -p slurm_logs

    ####### PREPARE ENV #######
    echo "Loading modules from $SLURM_PRESCRIPT_ML"
    source $SLURM_PRESCRIPT_ML
    ########################################
else 
    echo "Running on local machine"
    pref=''
fi

$pref python trainer_e2e.py --config config_train.yaml