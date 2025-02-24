#!/bin/bash
#SBATCH -J Dyn_Surr
### number of cores
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p wom
##SBATCH -B 1:8:2
#SBATCH --mem=20G
##SBATCH --gpus=1
#SBATCH --gres=shard:4
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o /home/ge45daw/slurm_logs/slurm-%J.out
#SBATCH -e /home/ge45daw/slurm_logs/slurm-%J.err

####### CLEANING THE ENV #######
module purge
################################

####### LOADING REQUIRED MODULES #######
module load spack_x86_64_v3
module load python/3.9-torch2-cuda12
########################################

####### LOADING REQUIRED ENVIRONMENTS #######
if test -d .venv; then
  source .venv/bin/activate
fi
#############################################


python engine.py    \
                    --outfolder "sd_train_ala2_flatemb_100ps" \
                    --propagator TFT \
                    --xtcfile ../ala2_100ps/md.xtc \
                    --tprfile ../ala2_100ps/md.tpr \
                    --resnames "ALA" "ACE" "NME" \
                    --tft_train_size 200 \
                    --tft_validation_size 20 \
                    --tft_seq_length 200 \
                    --encoder "FLATEMB" \
                    --nepochs 5000 \
                    --output_to_file \
                    --outpath . \
                    --save_checkpoint \
                    --lr 0.1 \
                    --scheduler \
                    # --enc_ckpt ./collective_encoder/run_ala2_100ps_5/EDVAE_checkpoint \

