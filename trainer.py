import sys
import os
import argparse

import torch

import wandb

from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

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
    parser.add_argument('--dynamics', type=str, default='XTC',
                        help='Type of dynamics data', choices=['XTC'])
    parser.add_argument('--nepochs', type=int, help='Number of epochs to run')

    # Output Settings
    parser.add_argument('--outpath', required=True, type=str,
                        help='Output folder for saving the training output')
    parser.add_argument('--outfolder', type=str, default='sd_training',
                        help='Stem of the folder name to save the output')
    parser.add_argument('--nexp', required=False, default=1,
                        type=int, help='Experiment number for output names')
    parser.add_argument('--wand', action="store_true",
                        help='Log to WandB logger')
    parser.add_argument('--tblogger', action="store_true",
                        help='Log to Tensorboard logger')
    parser.add_argument('--overwrite', action="store_true",
                        help='Overwrite output folder')
    parser.add_argument('--output_to_file', action="store_true",
                        help='Also store output in a file')

    # Save and/or Load Model
    parser.add_argument('--save_checkpoint',
                        action="store_true", help='Save Checkpoint')
    parser.add_argument('--load_model', default=None,
                        type=str, help='Load model from checkpoint')

    # Run parameters
    parser.add_argument('--nogpu', action="store_true",
                        help='Do not use gpu acceleration')
    parser.add_argument('--debug', action="store_true",
                        help='Run in debug mode')
    

    parser.add_argument('--lrate', type=float, default=1e-4,
                        help='Learning rate for the training')
    parser.add_argument('--scheduler', action="store_true",
                        help='Use learning rate scheduler')
    parser.add_argument('--nolog', action="store_true",
                        help='Dont log to wandb')
    # parser.add_argument('--l2norm', type=float, default=1e-3, help='Weights regularization for the training')
    # parser.add_argument('--nobatchnorm', action="store_false", help='Disable batch normalization in the network')

    args, _ = parser.parse_known_args()

    return args


args = parse_args()

if args.debug:
    args.nepochs = 2
    args.nexp = 0
    args.outfolder = "debug"
    args.overwrite = True
    args.nolog = True
    args.wand = False
    print("Running in debug mode")

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
# Creating Dataset
##################################

data_nested_args = vars(data_nested_args())
data_set = main_dl(**data_nested_args)

##################################
# Initilizing Surrogate Propagator
##################################

dyn_surrogate_args = {} | vars(dyn_surrogate_args())
prop_args = {}

if args.propagator == "TFT":
    dyn_surrogate_args["input_chunk_length"] = 5
    dyn_surrogate_args["output_chunk_length"] = 5
    dyn_surrogate_args["hidden_size"] = 64
    dyn_surrogate_args["lstm_layers"] = 1
    dyn_surrogate_args["num_attention_heads"] = 2
    dyn_surrogate_args["dropout"] = 0.1
    # dyn_surrogate_args["full_attention"] = True
    dyn_surrogate_args["batch_size"] = 50

    prop_args['n_epochs'] = args.nepochs
    # Marco: this needs to be true becaue i do not have covariant data
    prop_args['add_relative_index'] = True
    prop_args['add_encoders'] = None
    prop_args['optimizer_kwargs'] = {"lr": args.lrate}
    if args.scheduler:        
        prop_args['lr_scheduler_cls'] = torch.optim.lr_scheduler.ReduceLROnPlateau
        prop_args['lr_scheduler_kwargs'] = {"mode": "min",                  
                                            "factor": 0.5,
                                            "patience": 20,
                                            "min_lr": 1e-10,
                                            "cooldown": 100,
                                            "verbose": True
                                            }
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    prop_args['model_name'] = args.outfolder
    prop_args['random_state'] = 42
    prop_args['force_reset'] = True
    prop_args['save_checkpoints'] = True
    prop_args['work_dir'] = odir_name

    likelihood_args = {}
    # likelihood_args['quantiles'] = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    #                                 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 
    #                                 0.9, 0.95, 0.99]
    likelihood_args['quantiles'] = [0.25, 0.5, 0.75]
    # likelihood_args['enc_ckpt'] = args.enc_ckpt
    # likelihood_args['elems'] = data_set.loatn
    # likelihood_args['bond_connections'] = data_set.bonds
    
    # q = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
    #      0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    prop_args['likelihood'] = QuantileRegression(**likelihood_args)

    #########################################
    # Setup wandb logger
    #########################################
    wandb_config = {}
    for key, value in vars(args).items():
        wandb_config[key] = value
    #########################################
    earty_stop = EarlyStopping(monitor='val_loss',
                               mode='min',
                               patience=500,
                               min_delta=0.0001,
                               verbose=True,
                               )
    if args.nolog:
        logger = None
    else:
        logger = WandbLogger(project="DynSurrogate",
                             name=odir_name.strip('.').strip('/').replace('/','_'), config=wandb_config)

    prop_args['pl_trainer_kwargs'] = {
        "accelerator": 'auto',
        "devices": 'auto',
        "logger": logger,
        "callbacks": [lr_monitor],
    }
    
    if not args.nolog:
        logger.experiment.config.update(dyn_surrogate_args)
        logger.experiment.config.update(prop_args)
        logger.experiment.config.update(likelihood_args)


dyn_surrogate_args = prop_args | dyn_surrogate_args
prop_model = dyn_surrogate(**dyn_surrogate_args)


#########################################
# Training the Surrogate Propagator
# \
# prop_model.fit(series=train_series, past_covariates=train_series,
#                val_series=val_series, val_past_covariates=val_series, verbose=True)
prop_model.fit(series=data_set.get_train_series(), val_series=data_set.get_val_series(), verbose=True)
if not args.nolog:
    wandb.finish()
#########################################
# Saving the best model
#########################################
best_model = prop_model.load_from_checkpoint(
    model_name=prop_args['model_name'], work_dir=odir_name, best=True)
best_model.save(output_file_stem + "best.ckpt")
