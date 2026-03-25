import sys
import os
import shutil
import argparse
import yaml
import warnings

import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from utils_sd import *
sys.path.append(os.path.dirname(os.getcwd() + '/collective_encoder/'))
sys.path.append(os.path.dirname(os.getcwd() + '/propagators/'))

from embeddings.resolver import get_encdec


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
    config['wandb'] = False
    config['output_to_file'] = True
    
    config['encoder_args']['hidden_dim'] = 8
    config['encoder_args']['num_layers'] = 2
    
    config['decoder_args']['hidden_dim'] = 8
    config['decoder_args']['num_layers'] = 2

    config['propagator_args']['input_chunk_length'] = 2
    config['propagator_args']['output_chunk_length'] = 2
    config['propagator_args']['hidden_dim'] = 8

    config['data_args']['num_workers'] = 1
    config['data_args']['train_size'] = 2
    config['data_args']['validation_size'] = 2
    config['data_args']['batch_size'] = 2
    config['data_args']['val_batch_size'] = 2
    config['data_args']['n_seq_per_sample'] = 8
    print("Running in debug mode")

##################################
# Importing Lightning Modules
##################################
if config['propagator_name'] == "TFT":
    from propagators.tft import PropagatorTFT as dyn_surrogate
elif config['propagator_name'] == "BGE_TFT":
    from propagators.bge_tft import BondGraphEncoderTFT as dyn_surrogate
elif config['propagator_name'] == "BGE_TFT_V2":
    from propagators.bge_tft_v2 import BondGraphEncoderTFT as dyn_surrogate
else:
    raise ValueError("Unknown propagator type: " + config['propagator_name'])

if config['dynamics_name'] == 'XTC_graph':
    from dataloaders.sequence import SequenceDatamodule as main_dl
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
input_chunk_length = config['propagator_args']['input_chunk_length']
output_chunk_length = config['propagator_args']['output_chunk_length']
n_seq_per_sample = config['data_args'].pop('n_seq_per_sample', 1)
config['data_args']['input_chunk_length'] = input_chunk_length
config['data_args']['output_chunk_length'] = output_chunk_length
config['data_args']['num_chunks_per_sequence'] = n_seq_per_sample
config['data_args']['datareader_type'] = config['data_args'].get('datareader_type', 'XTC_CHUNKS')
config['data_args']['datareader_args'] = {
    'xtcfile': config['data_args'].pop('xtcfile', None),
    'tprfile': config['data_args'].pop('tprfile', None),
    'selection': config['data_args'].pop('selection', None),
}
if config['data_args']['datareader_type'] == 'XTC_CHUNKS_CG':
    config['data_args']['datareader_args']['cg_window'] = config['data_args'].pop('cg_window', 1)
data_set = main_dl(**config['data_args'])

if config.get('data_analyser', None):
    if config['data_analyser'] == 'ala2':
        from collective_encoder.plotters.ala2 import Ala2DataAnalyser as DataAnalyser
    else:
        warnings.warn("Unknown data analyser type: "+config['data_analyser'])

    analyser = DataAnalyser(output_dir=odir_name+"/data_analysis", data_args=config['data_args'])
    analyser.write_data(data_set.get_dataset())

##################################
# Initilizing Surrogate Propagator
##################################

# Setup PL callbacks
cbs = []

lr_monitor = LearningRateMonitor(logging_interval='epoch')
cbs.append(lr_monitor)

if config.get('early_stopping', True):
    early_stop = EarlyStopping(monitor='val_loss',
                            mode='min',
                            patience=config.get('early_stopping_args', {}).get('patience', 100),
                            min_delta=config.get('early_stopping_args', {}).get('min_delta', 1e-8),
                            verbose=True
                            )
    cbs.append(early_stop)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=odir_name + '/checkpoints',
    filename=config['propagator_name'] + '-{epoch:02d}-{val_loss:.6f}',
    save_top_k=1,
    mode='min',
)
cbs.append(checkpoint_callback)

# Setup wandb logger
if config.get('wandb', False):
    logger = WandbLogger(
                        project=config['wandb_project'],
                        entity=config['wandb_entity'],
                        name=odir_name.strip('.').strip('/').replace('/','_'), 
                        config=config,
                        save_dir=odir_name
                        )
else:
    logger = None

prop_args = {}
trainer_args = {}

if config['propagator_name'] == "TFT":
    encdec_name = config.get('encdec_name', None)
    encdec_ckpt = config.get('encdec_ckpt', None)
    if any(x is None for x in [encdec_name, encdec_ckpt]):
        raise ValueError("Encoder name and checkpoint must be specified for TFT propagator")
    encdec_cls = get_encdec(encdec_name)
    encdec = encdec_cls.load_from_checkpoint(encdec_ckpt, datamodule=data_set)
    prop_args['encdec_model'] = encdec
elif config['propagator_name'] in ["BGE_TFT", "BGE_TFT_V2"]:
    prop_args['datamodule'] = data_set
    prop_args['encoder_args'] = config['encoder_args']
    prop_args['decoder_args'] = config['decoder_args']
    prop_args['loss_prop_weight'] = config['loss_prop_weight']
    prop_args['loss_prop_start'] = config.get('loss_prop_start', 0)
    prop_args['loss_rec_weight'] = config['loss_rec_weight']
    prop_args['loss_e2e_weight'] = config['loss_e2e_weight']
    prop_args['loss_encdec_weights'] = config['loss_encdec_weights']
else:
    raise ValueError("Unknown propagator type")

prop_args['propagator_args'] = config['propagator_args']

prop_args['likelihood'] = config.get('likelihood', 'QuantileRegression')
prop_args['likelihood_args'] = config.get('likelihood_args', None)

prop_args['lr'] = config.get('lrate', 1e-3)
prop_args['weight_decay'] = config.get('weight_decay', 0.0) 
prop_args['normIn'] = config.get('normIn', False)
prop_args['scheduler'] = config.get('scheduler', None)
prop_args['scheduler_args'] = config.get('scheduler_args', None)
prop_args['outname'] = output_file_stem

trainer_args['accelerator'] = 'auto'
trainer_args['devices'] = 'auto'
trainer_args['logger'] = logger
trainer_args['callbacks'] = cbs

if config.get('wandb', False):
    logger.experiment.config.update(prop_args)

if config.get('load_ckpt', None) is not None:
    print("Loading model from checkpoint: " + config['load_ckpt'])
    model = dyn_surrogate.load_from_checkpoint(config['load_ckpt'], **prop_args)
else:
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

if config.get('wandb', False):
    wandb.finish()

#########################################
# Saving the best model
#########################################
shutil.copyfile(checkpoint_callback.best_model_path, odir_name + "/checkpoints/best.ckpt")
print("Best model saved at " + odir_name + "/checkpoints/best.ckpt")
