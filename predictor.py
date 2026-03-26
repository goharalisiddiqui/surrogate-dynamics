import os
import sys
import argparse
import yaml

from projectutils import *

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
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small data and epochs')
    
    args = parser.parse_args()

    return args


args = parse_args()

assert os.path.isfile(args.config), "Config file not found!"
config = yaml.safe_load(open(args.config, 'r'))
if args.debug:
    config['debug'] = True

if config['debug']:
    config['nexp'] = 0
    config['outpath'] = "predict_runs"
    config['outfolder'] = "debug"
    config['overwrite'] = True
    config['nolog'] = True
    config['output_to_file'] = True

    config['data_args']['predict_steps'] = 100
    print("Running in debug mode")

##################################
# Importing Lightning Modules
##################################
if config['propagator_name'] == "TFT":
    from propagators.tft import PropagatorTFT as PropagatorModel
elif config['propagator_name'] == "BGE_TFT":
    from propagators.bge_tft import BondGraphEncoderTFT as PropagatorModel
else:
    raise ValueError("Unknown propagator type")

if config['dynamics_name'] == 'XTC_latent':
    from dataloaders.xtc_latent import XtcTrainer as DataModule
elif config['dynamics_name'] == 'XTC_graph':
    from collective_encoder.dataloaders.default import DefaultDatamodule as DataModule
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

# Check beforehand if the checkpoint file exists
assert os.path.isfile(config['propagator_ckpt']), "Propagator model path does not exist"

##################################
# Creating Dataset
##################################
datamod_args = config['data_args'].copy()
# Peek in the model to get the required input length
hpr = torch.load(config['propagator_ckpt'], map_location='cpu', weights_only=False)['hyper_parameters']
input_chunk_length = hpr['propagator_args']['input_chunk_length']
output_chunk_length = hpr['propagator_args']['output_chunk_length']
predict_steps = datamod_args.pop('predict_steps', None)
if predict_steps is None:
    raise ValueError("predict_steps must be specified in data_args in the config file")
if predict_steps < input_chunk_length:
    raise ValueError("predict_steps must be greater than or equal to input_chunk_length")
if config.get('plot_decoded', False):
    datamod_args['train_size'] = predict_steps
else:
    datamod_args['train_size'] = input_chunk_length
datamod_args['batch_size'] = datamod_args['train_size']
datamod_args['sequential'] = True
datamod = DataModule(**datamod_args)

##################################
# Loding trained propagator model
##################################
print("Loading propagator model from: ", config['propagator_ckpt'])
try:
    prop = PropagatorModel.load_from_checkpoint(config['propagator_ckpt'], datamodule=datamod)
except Exception as e:
    raise RuntimeError(f"Error loading the propagator model from checkpoint \n"
                       "Check if the model architecture matches the checkpoint.")
# print("Model details:")
# for name, param in prop.hparams.items():
#     if isinstance(param, dict):
#         print(f"{name}:")
#         for k, v in param.items():
#             print(f"    {k}: {v}")
#     else:
#         print(f"{name}: {param}")
# print("=========================================\n\n")



##################################
# Prediction
##################################
pred_writer = PredictionWriter(output_dir=odir_name, write_interval="epoch", config=config.get('writer_args', {}))
predictor_args = {}
predictor_args['default_root_dir'] = odir_name
predictor_args['accelerator'] = "cpu"
predictor_args['devices'] = 1
predictor_args['callbacks'] = [pred_writer]

prop.set_predict_settings(
    predict_steps=config['data_args']['predict_steps'],
    sampling_temperature=config.get('sampling_temperature', 1.0)
)
trainer = pl.Trainer(**predictor_args)

trainer.predict(prop, datamodule=datamod, return_predictions=False)
