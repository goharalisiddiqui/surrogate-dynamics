import logging
import os
import wandb
import yaml
import shutil

import argparse
import warnings

_log = logging.getLogger(__name__)

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from gslibs.utils.common import recursive_update
from gslibs.utils.common import get_required_init_args

from propagators.resolver import get_propagator
from collective_encoder.datamodules.resolver import get_datamodule
from collective_encoder.common.config_check import (
    validate_duplicate_keys, 
    validate_required_fields 
)
from collective_encoder.dataanalysers.resolver import get_dataanalyser
from gslibs.utils.filesystem import create_rundir, output_to_file

warnings.filterwarnings("ignore", ".*does not have many workers.*")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'trainer', 'defaults.yaml')
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'trainer', 'debug.yaml')
OVERRIDABLE_DMOD_ARGS = ['batch_size', 'val_batch_size', 'num_workers']
torch.set_default_dtype(torch.float64)
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

def train(config_path: str, debug: bool = False):
    """Train a collective encoder model based on the provided configuration."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    validate_duplicate_keys(config_path)
    config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, 'r'))
    recursive_update(config, yaml.safe_load(open(config_path, 'r')))
    if debug or config.get('debug', False):
        # Load debug config and override values
        recursive_update(config, yaml.safe_load(open(DEBUG_CONFIG_PATH, 'r')))
        torch.manual_seed(0)
        np.random.seed(0)
        print("Running in debug mode.")
    validate_required_fields(config)

    ##################################
    # Output directory
    ##################################
    run_dir = create_rundir(config['outpath'], 
                        config['outfolder'], 
                        config['nexp'], 
                        overwrite=config['overwrite'])

    ##################################
    # Output to file
    ##################################
    if config['output_to_file']:
        output_to_file(run_dir, filename="out.txt")
    
    ##################################
    # Meta args used in all modules
    ##################################
    logging.basicConfig(filename=os.path.join(run_dir, "run.log"),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    metargs = {
        'verbose': config.get('verbose', True),
        'root_logger_name': __name__,
        'run_dir': run_dir,
    }
    
    ##################################
    # Creating Dataset
    ##################################
    if 'load_datamodule' in config:
        dmod_path = config['load_datamodule']
        _log.info("Loading datamodule from: " + dmod_path)
        dmod_ckpt = os.path.join(dmod_path, "datamodule.pth")
        
        # torch.serialization.add_safe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(dmod_ckpt)) # !!! Very Unsafe, only do this if you trust the source of the checkpoint !!!
        dm = torch.load(dmod_ckpt, weights_only=False)
        dm_override_args = config.get('datamodule_args', {})
        if not hasattr(dm, "args"): # Backwards compatibility for datamodules that do not have get_args method. Will use hparams if available, otherwise empty dict.
            _log.warning("Loaded datamodule does not have 'get_args' method.")
            dm_args = dm.hparams if hasattr(dm, "hparams") else {}
        else:
            dm_args = dm.get_args()
        for key, value in dm_override_args.items():
            if key not in OVERRIDABLE_DMOD_ARGS:
                raise ValueError(f"Cannot override datamodule argument '{key}'. "
                                 f"Allowed keys: {OVERRIDABLE_DMOD_ARGS}")
            _log.info(f"Overriding datamodule argument '{key}' with "
                      f"value: {value}, previous value: {getattr(dm, key, 'N/A')}")
            dm_args[key] = value
            setattr(dm, key, value)
    else:
        dm_type = config['datamodule_type']
        dm_args = config['datamodule_args']
        dm_cls = get_datamodule(dm_type)
        validate_required_fields(dm_args, 
                                get_required_init_args(dm_cls))
        
        dataset_type = dm_args.get('dataset_type', None)
        if dataset_type not in nn_cls._COMPATIBLE_DATASETS:
            raise ValueError(f"Network '{nn_type}' is not compatible with dataset" \
                            f" '{dataset_type}'")
        dm = dm_cls(dm_args, **metargs)
        
    ##################################
    # Data analysis and visualization
    ##################################
    da = config.get('data_analyser_type', None)
    if da != None:
        analyser_cls = get_dataanalyser(da)    
        da_args = config.get('data_analyser_args', {})
        da_args['datamodule_args'] = dm_args
        da_args['output_dir'] = run_dir + "/data_analysis"
        analyser = analyser_cls(da_args,**metargs)
        analyser.write_data(dm.get_train_dataset(), label="train")
        analyser.write_data(dm.get_val_dataset(), label="val")

    ##################################
    # Setting up the NN
    ##################################
    nn_type = config['propagator_type']
    nn_cls = get_propagator(nn_type)
    req_fields = get_required_init_args(nn_cls)
    validate_required_fields(config['propagator_args'], req_fields)
    nn_args = {
        'lrate': config['lrate'],
        'weight_decay': config['weight_decay'],
        'normIn': config['normIn'],
        'scheduler': config['scheduler'],
        'scheduler_args': config.get('scheduler_args', {}),
    }
    nn_args.update(config.get('propagator_args', {}))

    load_model = config.get('load_model', None)
    if load_model != None:
        checkpoint_file = load_model
        print(f"Loading model from {checkpoint_file}")
        model = nn_cls.load_from_checkpoint(checkpoint_file, 
                                            datamodule=dm,
                                            args=nn_args, **metargs)
    else:
        model = nn_cls(datamodule=dm, 
                       args=nn_args, 
                       **metargs)

    ##################################
    # Training the NN
    ##################################
    trainargs = {"max_epochs" : config['nepochs'],
                 "log_every_n_steps" : 1,
                 "default_root_dir" : run_dir}
    if not config.get('nogpu', False):
        trainargs["accelerator"] = 'auto'
        trainargs["devices"] = 'auto'
    if config.get('wandb', False):
        wandb_logger = WandbLogger(project=config['wandb_project'],
                                 entity=config['wandb_entity'],
                                 save_dir=run_dir,
                                 name=run_dir.strip(".").strip("/").replace("/", "_"),
                                 log_model=False,)
        # wandb_logger.watch(model, log_graph=True)
        trainargs["logger"] = wandb_logger

    callbacks = []
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    # Early stopping
    if config.get('early_stopping', True):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_args', {}).get('patience', 100),
            min_delta=config.get('early_stopping_args', {}).get('min_delta', 1e-8),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=run_dir + '/checkpoints',
        filename=config['propagator_type'] + '-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        mode='min',
    )
    callbacks.append(checkpoint_callback)

    trainargs["callbacks"] = callbacks
    # trainargs["num_sanity_val_steps"] = 0

    # trainargs["gradient_clip_val"] = 0.5
    # trainargs["gradient_clip_algorithm"] = "norm"

    trainer = pl.Trainer(**trainargs)

    if config['nepochs'] > 0:
        trainer.fit(model, datamodule=dm)
        if config.get('wandb', False):
            wandb.finish()

    if config['nepochs'] == 0 and load_model == None:
        _log.warning("Both nepochs and load_model are not set. Nothing to do.")

    # Save the best model checkpoint as best.ckpt
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path != "":
        shutil.copy(best_checkpoint_path, os.path.dirname(best_checkpoint_path) + "/best.ckpt")
    _log.info(f"Best model saved at: {best_checkpoint_path}")


    ##################################
    if config.get('test_plotter_type', False):
        model.add_test_plotter(config['test_plotter_type'], config.get('test_plotter_args', None))  
        trainer.test(model, datamodule=dm)


def main():
    """Main entry point for the collective encoder training."""
    args = parse_args()
    train(args.config, args.debug)

if __name__ == "__main__":
    main()





