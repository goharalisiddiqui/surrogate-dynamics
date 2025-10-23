#! /bin/bash
#SBATCH -J TFT_PRED
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpucloud
#SBATCH --mem=12G
#SBATCH --gres=shard:3
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

#Read command line arguments
while getopts d flag
do
    case "${flag}" in
        d) debug=1;;
    esac
done

if [ "$debug" == 1 ]; then
    $pref python predictor.py --config config_predict.yaml --debug
else
    $pref python predictor.py --config config_predict.yaml
fi
