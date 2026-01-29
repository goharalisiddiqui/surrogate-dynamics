#! /bin/bash
#SBATCH -J TFT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpucloud
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_4000_ada:1
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err


####### PREPARE ENV #######
echo "Loading modules from $SLURM_PRESCRIPT_ML"
source $SLURM_PRESCRIPT_ML
########################################


unset $pref
if [ ! -z "${SLURM_JOB_ID}" ]; then
    echo "Running on compute node"
    pref='srun'
    mkdir -p slurm_logs

else 
    echo "Running on local machine"
    pref=''
fi

#Read command line arguments
while getopts d flag
do
    case "${flag}" in
        d) debug=1;;
    esac
done

if [ "$debug" == 1 ]; then
    $pref python trainer.py --config config_train_propagator.yaml --debug
else
    $pref python trainer.py --config config_train_propagator.yaml
fi