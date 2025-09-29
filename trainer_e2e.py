import sys
import os
import shutil
import argparse
import yaml

import torch

import wandb

from darts.utils.likelihood_models import QuantileRegression

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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
    parser.add_argument('--config', required=True, type=str,
                        help='')
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
    config['nepochs'] = 3
    config['nexp'] = 0
    config['outpath'] = "train_runs"
    config['outfolder'] = "debug"
    config['overwrite'] = True
    config['nolog'] = True
    config['wand'] = False
    config['output_to_file'] = True

    config['propagator_args']['input_chunk_length'] = 2
    config['propagator_args']['output_chunk_length'] = 2

    config['data_args']['num_workers'] = 1
    config['data_args']['train_size'] = 10
    config['data_args']['validation_size'] = 6
    config['data_args']['batch_size'] = 2
    config['data_args']['val_batch_size'] = 2
    print("Running in debug mode")

##################################
# Importing Lightning Modules
##################################
if config['propagator_name'] == "TFT":
    from propagators.tft_net import TFTModel as dyn_surrogate
    from propagators.tft_net import TFT_args as dyn_surrogate_args
elif config['propagator_name'] == "BGE_TFT":
    from propagators.bge_tft import BondGraphEncoderTFT as dyn_surrogate
else:
    raise ValueError("Unknown propagator type: " + config['propagator_name'])

if config['dynamics_name'] == 'XTC':
    from dataloaders.xtc_graph import XtcSequence as main_dl
else:
    raise ValueError("Unknown data type: " + config['dynamics_name'])

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
config['data_args']['sequence_length'] = (
    config['propagator_args']['input_chunk_length'] +
    config['propagator_args']['output_chunk_length']
)
data_set = main_dl(**config['data_args'])

##################################
# Initilizing Surrogate Propagator
##################################

# Setup PL callbacks
lr_monitor = LearningRateMonitor(logging_interval='epoch')
early_stop = EarlyStopping(monitor='val_loss',
                           mode='min',
                           patience=100,
                           min_delta=1e-8,
                           verbose=True,
                           )
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=odir_name + '/checkpoints',
    filename=config['propagator_name'] + '-{epoch:02d}-{val_loss:.6f}',
    save_top_k=1,
    mode='min',
)
cbs = [lr_monitor, checkpoint_callback, early_stop]

# Setup wandb logger
if config['nolog']:
    logger = None
else:
    logger = WandbLogger(
                        project=config['wand_project'],
                        entity=config['wand_entity'],
                        name=odir_name.strip('.').strip('/').replace('/','_'), 
                        config=config,
                        save_dir=odir_name
                        )

prop_args = {}
trainer_args = {}
if config['propagator_name'] == "TFT":
    prop_args = config['propagator_args']
    prop_args["batch_size"] = config['batch_size']

    prop_args['n_epochs'] = config['nepochs']
    prop_args['add_relative_index'] = True
    prop_args['add_encoders'] = None
    prop_args['optimizer_kwargs'] = {"lr": config['lrate']}
    if config['scheduler']:        
        prop_args['lr_scheduler_cls'] = torch.optim.lr_scheduler.ReduceLROnPlateau
        prop_args['lr_scheduler_kwargs'] = {"mode": "min",                  
                                            "factor": 0.5,
                                            "patience": 20,
                                            "min_lr": 1e-10,
                                            "cooldown": 100,
                                            "verbose": True
                                            }
    

    prop_args['model_name'] = config['outfolder']
    prop_args['random_state'] = 42
    prop_args['force_reset'] = True
    prop_args['save_checkpoints'] = True
    prop_args['work_dir'] = odir_name

    likelihood_args = {}
    # likelihood_args['quantiles'] = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    #                                 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 
    #                                 0.9, 0.95, 0.99]
    likelihood_args['quantiles'] = [0.25, 0.5, 0.75]
    
    prop_args['likelihood'] = QuantileRegression(**likelihood_args)
    prop_args['pl_trainer_kwargs'] = {
        "accelerator": 'auto',
        "devices": 'auto',
        "logger": logger,
        "callbacks": cbs,
    }
    

elif config['propagator_name'] == "BGE_TFT":
    prop_args['datamodule'] = data_set
    prop_args['encoder_args'] = config['encoder_args']
    prop_args['decoder_args'] = config['decoder_args']
    prop_args['propagator_args'] = config['propagator_args']

    prop_args['quantiles'] = [0.25, 0.5, 0.75]

    prop_args['lr'] = config['lrate']
    prop_args['loss_prop_weight'] = config['loss_prop_weight']
    prop_args['weight_decay'] = config['weight_decay']
    prop_args['normIn'] = True
    prop_args['scheduler'] = config['scheduler']
    prop_args['outname'] = output_file_stem

    trainer_args['accelerator'] = 'auto'
    trainer_args['devices'] = 'auto'
    trainer_args['logger'] = logger
    trainer_args['callbacks'] = cbs


else:
    raise ValueError("Unknown propagator type")



if not config['nolog']:
    logger.experiment.config.update(prop_args)


model = dyn_surrogate(**prop_args)


#########################################
# Training the Surrogate Propagator
#########################################
trainer_args['max_epochs'] = config['nepochs']
trainer_args['log_every_n_steps'] = 1
trainer_args['default_root_dir'] = odir_name
trainer_args['num_sanity_val_steps'] = 0
trainer = pl.Trainer(**trainer_args)

trainer.fit(model, datamodule=data_set)

if not config['nolog']:
    wandb.finish()

#########################################
# Saving the best model
#########################################
shutil.copyfile(checkpoint_callback.best_model_path, odir_name + "/checkpoints/best.ckpt")
print("Best model saved at " + odir_name + "/checkpoints/best.ckpt")
