python engine.py    \
                    --propagator TFT \
                    --xtcfile ../ala2_100ns/md.xtc \
                    --tprfile ../ala2_100ns/md.tpr \
                    --resname "ALA" \
                    --datasize 100 \
                    --encoder "EDVAE" \
                    --enc_ckpt ./collective_encoder/run_ala2_100ns_1/EDVAE_checkpoint \
                    --nepochs 10 \
                    --output_to_file \
                    --outpath . \
                    --overwrite \
                    --save_checkpoint \
                    --lr 0.01 \
                    --scheduler \

