#! /bin/bash
#SBATCH -J TFT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpucloud
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

source slurm_header.sh

if [ "$debug" == 1 ]; then
    $pref python trainer.py --config config_train_e2e.yaml --debug
else
    $pref python trainer.py --config config_train_e2e.yaml
fi