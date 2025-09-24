import os
import sys
import argparse

import torch

from darts import TimeSeries

from utils_sd import *


sys.path.append(os.path.dirname(os.getcwd() + '/collective_encoder/'))
sys.path.append(os.path.dirname(os.getcwd() + '/propagators/'))


##################################
# Arguments
##################################
def parse_args():
    desc = "Surrogate model to predict dynamics of molecular systems as time series data"
    parser = argparse.ArgumentParser(description=desc)

    # Run Settings
    parser.add_argument('--propagator', type=str, default='TFT',
                        help='Type of propagator', choices=['TFT'])
    parser.add_argument('--encoder', type=str, default='EDVAE',
                        help='Type of encoder', choices=['EDVAE', 'EDVAEGAN', 'FLATEMB'])
    parser.add_argument('--dynamics', type=str, default='XTC',
                        help='Type of dynamics data', choices=['XTC'])
    parser.add_argument('--enc_ckpt', default=None,
                        type=str, help='Saved PL module of encoder')
    parser.add_argument('--prop_model', required=True,
                        type=str, help='Saved PL model name of propagator')

    # Output Settings
    parser.add_argument('--outpath', required=True, type=str,
                        help='Output folder for saving the training output')
    parser.add_argument('--outfolder', type=str, default='sd_prediction',
                        help='Stem of the folder name to save the output')
    parser.add_argument('--nexp', required=False, default=1,
                        type=int, help='Experiment number for output names')
    parser.add_argument('--overwrite', action="store_true",
                        help='Overwrite output folder')
    parser.add_argument('--output_to_file', action="store_true",
                        help='Also store output in a file')

    # Prediction settings
    parser.add_argument('--n_windows', type=int, default=1,
                        help='How many prediction windows to run')

    # Save and/or Load Model
    # parser.add_argument('--save_checkpoint',
    #                     action="store_true", help='Save Checkpoint')
    # parser.add_argument('--load_model', default=None,
    #                     type=str, help='Load model from checkpoint')

    # Run parameters
    parser.add_argument('--nogpu', action="store_true",
                        help='Do not use gpu acceleration')

    args, _ = parser.parse_known_args()

    return args


args = parse_args()

##################################
# Importing Lightning Modules
##################################
if args.propagator == "TFT":
    from propagators.tft_net import TFTModel as dyn_surrogate
    from propagators.tft_net import TFT_args as dyn_surrogate_args
    # from propagators.tft_net import ModifiedQuantileRegression as QuantileRegression
else:
    raise ValueError("Unknown propagator type")

if args.dynamics == 'XTC':
    from dataloaders.xtc_trainer import XtcTrainer as main_dl
    from dataloaders.xtc_trainer import XTCT_args as data_nested_args
else:
    raise ValueError("Unknown data type")

##################################
# Output directory
##################################
odir = args.outpath + "/" + args.outfolder + "_"
nexp = args.nexp
odir_name = odir+str(nexp)
if not args.overwrite:
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
output_file_stem = odir_name+"/"+args.propagator+"_"


##################################
# Output to file
##################################
if args.output_to_file:
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
# Loding trained propagator model
##################################
assert os.path.isdir(args.prop_model), "Propagator model path does not exist"
prop_name = "_".join(args.prop_model.split('_')[:-1])
prop_name = os.path.basename(prop_name)
prop = dyn_surrogate.load_from_checkpoint(
    model_name=prop_name, work_dir=args.prop_model, best=True)
print("Loaded propagator model from: ", args.prop_model)
# Disable wandb logging since it fails on predict
prop.trainer_params['logger'] = None


##################################
# Creating Dataset
##################################
n_predict_windows = args.n_windows
nsteps_warmup = prop.input_chunk_length
nsteps_to_predict = prop.output_chunk_length
print(f"Using {nsteps_warmup} warmup steps and {nsteps_to_predict} prediction steps")

data_nested_args = vars(data_nested_args())
data_nested_args['series_length'] = nsteps_warmup + nsteps_to_predict
data_nested_args['train_size'] = 0
data_nested_args['val_size'] = n_predict_windows
data_nested_args['verbose'] = True
pdata = main_dl(**data_nested_args)


##################################
# Prediction
##################################
warmup_steps = pdata.get_warmup_steps(nsteps_warmup)
print(f"Using {warmup_steps.shape[0]} warmup steps for prediction")
for i in range(n_predict_windows):
    print(f"Predicting window {i+1}/{n_predict_windows}")
    warmup_series = TimeSeries.from_values(warmup_steps)
    predicted_steps = prop.predict(
        series=warmup_series, n=nsteps_to_predict, num_samples=1)
    predicted_steps = predicted_steps.all_values()
    predicted_steps = torch.tensor(predicted_steps)
    predicted_steps = torch.mean(predicted_steps, 2)

    pdata.add_predicted(predicted_steps)
    
    if i < n_predict_windows - 1:
        warmup_steps = pdata.get_warmup_steps(nsteps_warmup)
pdata.output_results(output_file_stem)
