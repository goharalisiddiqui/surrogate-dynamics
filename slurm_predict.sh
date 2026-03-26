#! /bin/bash
#SBATCH -J TFT_PRED
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p carlos
#SBATCH --mem=12G
#SBATCH --gpus=1
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

source slurm_header.sh

if [ "$debug" == 1 ]; then
    $pref python predictor.py --config config_predict.yaml --debug
else
    $pref python predictor.py --config config_predict.yaml
fi
