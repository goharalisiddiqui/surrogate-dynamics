import os
import sys
import argparse

import torch

from darts import TimeSeries

from utils import *


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
                        help='Type of encoder', choices=['EDVAE', 'FLATEMB'])
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
    parser.add_argument('--n_warmup', type=int, default=0,
                        help='Number of warmup steps to use for prediction')
    parser.add_argument('--n_predict', type=int, default=0,
                        help='Number of steps to predict')
    parser.add_argument('--n_windows', type=int, default=0,
                        help='Number of steps to predict')

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
else:
    raise ValueError("Unknown propagator type")
if args.encoder == "EDVAE":
    if args.enc_ckpt is None:
        raise ValueError("EDVAE encoding requires a path to the trained encoder checkpoint via --enc_ckpt option")
    from nets.edvae_net import EDVAE as enc_model
elif args.encoder == "FLATEMB":
    from embeddings.flatemb import FlatEmb as enc_model
else:
    raise ValueError("Unknown encoder type")

if args.dynamics == 'XTC':
    from data_predictors.xtc_predictor import XtcPredictor as PredictorData
    from data_predictors.xtc_predictor import XTCP_args as PredictorData_args
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
# Loding trained encoder model
##################################
if args.encoder == "EDVAE":
    assert args.enc_ckpt is not None, "EDVAE encoding requires a path to the trained encoder checkpoint via --enc_ckpt option"
    enc = enc_model.load_from_checkpoint(args.enc_ckpt)
    print("Loaded encoder model latent dimensionality: ", enc.dim_latent)
    # enc.metaD = True
elif args.encoder == "FLATEMB":
    enc = enc_model()
else:
    raise ValueError("Unknown encoder type")

##################################
# Loding trained propagator model
##################################
prop_name = "_".join(args.prop_model.split('_')[:-1])
prop = dyn_surrogate.load_from_checkpoint(
    model_name=prop_name, work_dir=args.prop_model, best=True)
##################################
# Creating Dataset
##################################
# Disable wandb logging since it fails on predict
prop.trainer_params['logger'] = None

n_predict_windows = args.n_windows
nsteps_warmup = args.n_warmup
nsteps_to_predict = args.n_predict
if nsteps_to_predict == 0:
    nsteps_to_predict = 100 # FIXME: Get output chunk size from TFT model
if nsteps_warmup == 0:
    Warning("No warmup steps provided. Using 20% of prediction steps as warmup")
    nsteps_warmup = 100 # FIXME: Get input chunk size from TFT model

data_nested_args = vars(PredictorData_args())
data_nested_args['verbose'] = True
data_nested_args['dataset_size'] = n_predict_windows * (nsteps_warmup + nsteps_to_predict)
pdata = PredictorData(**data_nested_args)

##################################
# Prediction
##################################
for i in range(n_predict_windows):
    print(f"Predicting window {i+1}/{n_predict_windows}")
    warmup_steps = pdata.get_warmup_steps(nsteps_warmup)
    encoded_warmup_steps = enc.get_latent_mean(warmup_steps)

    warmup_series = TimeSeries.from_values(encoded_warmup_steps)
    # predicted_steps = prop.predict(
    #     series=warmup_series, past_covariates=warmup_series, n=nsteps_to_predict, num_samples=100)
    predicted_steps = prop.predict(
        series=warmup_series, n=nsteps_to_predict, num_samples=100)
    predicted_steps = predicted_steps.all_values()
    predicted_steps = torch.tensor(predicted_steps)
    predicted_steps = torch.mean(predicted_steps, 2)

    predicted_steps = enc.decode_latent(predicted_steps)
    pdata.extend_trajectory(predicted_steps)
# np.set_printoptions(threshold=sys.maxsize)
# print(pdata.get_coordinates())

pdata.output_trajectory(output_file_stem)
pdata.output_real_trajectory(output_file_stem)
