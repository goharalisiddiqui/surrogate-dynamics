import os
import sys
import argparse
import yaml

from utils_sd import *

import pytorch_lightning as pl
sys.path.append(os.path.dirname(os.getcwd() + '/collective_encoder/'))
sys.path.append(os.path.dirname(os.getcwd() + '/propagators/'))


##################################
# Arguments
##################################

def parse_args():
    desc = "Surrogate model to predict dynamics of molecular systems as time series data"
    parser = argparse.ArgumentParser(description=desc)

    # Run Settings
    parser.add_argument('--config', required=True, type=str,
                        help='Config file for prediction')
    
    args = parser.parse_args()

    return args


args = parse_args()

assert os.path.isfile(args.config), "Config file not found!"
config = yaml.safe_load(open(args.config, 'r'))

if config['debug']:
    config['nexp'] = 0
    config['outpath'] = "."
    config['outfolder'] = "pred_debug"
    config['overwrite'] = True
    config['nolog'] = True
    config['output_to_file'] = True

    config['data_args']['predict_steps'] = 100
    print("Running in debug mode")

##################################
# Importing Lightning Modules
##################################
if config['propagator_name'] == "TFT":
    from propagators.tft_net import TFTModel as PropagatorModel
elif config['propagator_name'] == "BGE_TFT":
    from propagators.bge_tft import BondGraphEncoderTFT as PropagatorModel
else:
    raise ValueError("Unknown propagator type")

if config['dynamics_name'] == 'XTC_latent':
    from dataloaders.xtc_latent import XtcTrainer as DataModule
elif config['dynamics_name'] == 'XTC_graph':
    from dataloaders.xtc_graph import XtcSequence as DataModule
else:
    raise ValueError("Unknown data type")

if config['writer_name'] == 'ala2':
    from plotters.ala2 import Ala2Writer as PredictionWriter
else:
    PredictionWriter = None

##################################
# Output directory
##################################
odir = config['outpath'] + "/" + config['outfolder'] + "_"
nexp = config['nexp']
odir_name = odir+str(nexp)
if not config['overwrite']:
    while True:
        odir_name = odir+str(nexp)
        if not os.path.isdir(odir_name):
            os.makedirs(odir_name)
            break
        nexp = nexp + 1
else:
    if not os.path.isdir(odir_name):
        os.makedirs(odir_name)

if len(os.listdir(odir_name)) != 0:
    import shutil
    shutil.rmtree(odir_name, ignore_errors=True)
    os.mkdir(odir_name)
output_file_stem = odir_name+"/"+config['propagator_name']+"_"


##################################
# Output to file
##################################
if config['output_to_file']:
    import sys
    import subprocess
    print("Redirecting output to file "+odir_name+"/out.txt")
    tee = subprocess.Popen(
        ["tee", odir_name+"/out.txt"], stdin=subprocess.PIPE)
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())




##################################
# Creating Dataset
##################################
datamod_args = config['data_args']
datamod = DataModule(**datamod_args)

##################################
# Loding trained propagator model
##################################
assert os.path.isfile(config['propagator_ckpt']), "Propagator model path does not exist"
prop = PropagatorModel.load_from_checkpoint(config['propagator_ckpt'], datamodule=datamod)
print("Loaded propagator model from: ", config['propagator_ckpt'])


##################################
# Prediction
##################################
pred_writer = PredictionWriter(output_dir=odir_name, write_interval="epoch")
predictor_args = {}
predictor_args['default_root_dir'] = odir_name
predictor_args['accelerator'] = "cpu"
predictor_args['devices'] = 1
predictor_args['callbacks'] = [pred_writer]

trainer = pl.Trainer(**predictor_args)

trainer.predict(prop, datamodule=datamod, return_predictions=False)




# warmup_steps = pdata.get_warmup_steps(nsteps_warmup)
# print(f"Using {warmup_steps.shape[0]} warmup steps for prediction")
# for i in range(n_predict_windows):
#     print(f"Predicting window {i+1}/{n_predict_windows}")
#     warmup_series = TimeSeries.from_values(warmup_steps)
#     predicted_steps = prop.predict(
#         series=warmup_series, n=nsteps_to_predict, num_samples=1)
#     predicted_steps = predicted_steps.all_values()
#     predicted_steps = torch.tensor(predicted_steps)
#     predicted_steps = torch.mean(predicted_steps, 2)

#     pdata.add_predicted(predicted_steps)
    
#     if i < n_predict_windows - 1:
#         warmup_steps = pdata.get_warmup_steps(nsteps_warmup)
# pdata.output_results(output_file_stem)
