#! /bin/bash
#SBATCH -J CE
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpucloud
#SBATCH --mem=32G
#SBATCH --gres=shard:8
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

                    # --datasize 50 --sequential \
$pref python collective_encoder/engine.py \
                    --datatype XTC \
                    --xtcfile ./data_generation/ala2/ala2_100ns/md.xtc \
                    --tprfile ./data_generation/ala2/ala2_100ns/md.tpr \
                    --selection "(resname ALA or resname ACE or resname NME) and not element H" \
                    --dataset GRAPH \
                    --norm_type standard \
                    --batch_size 8 \
                    --train_prop 0.8 \
                    --validation_prop 0.1 \
                    --val_batch_size 9 \
                    --verbose \
                    --outpath ./run_BGE \
                    --outfolder "run" --nexp 4 --overwrite \
                    --wandb \
                    --wandb_project "GraphEncoder" \
                    --output_to_file \
                    --save_checkpoint \
                    --networktype "GRAPH_ENCODER" \
                    --nepochs 100 \
                    --enc_node_embed_dim 10 \
                    --enc_edge_embed_dim 2 \
                    --enc_hidden_dim 128 \
                    --enc_num_layers 5 \
                    --enc_heads 8 \
                    --set2set_steps 3 \
                    --enc_dropout 0.0 \
                    --latent_dim 32 \
                    --template_khop 2 \
                    --dec_node_embed_dim 10 \
                    --dec_edge_embed_dim 2 \
                    --dec_hidden_dim 128 \
                    --dec_num_layers 5 \
                    --dec_heads 4 \
                    --dec_dropou0.0 \
                    --dec_dropout_mlp 0.1 \
                    --lr 0.0001 \
                    --weight_decay 0.001 \
                    --normalize_inputs \
                    --scheduler \