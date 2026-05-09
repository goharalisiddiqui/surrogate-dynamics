import logging
from pyexpat import model
_log = logging.getLogger(__name__)

import os
import yaml
import argparse
import warnings

import numpy as np

import torch
import pytorch_lightning as pl

from gslibs.utils.common import recursive_update
from gslibs.utils.common import get_required_init_args
from gslibs.utils.filesystem import create_rundir, output_to_file

from collective_encoder.utils import check_dict_contains_keys
from propagators.resolver import get_propagator
from collective_encoder.common.config_check import (
    validate_duplicate_keys, 
    validate_required_fields 
)
from collective_encoder.dataanalysers.resolver import get_dataanalyser

from datamodules.resolver import get_datamodule
from inferenceplotters.resolver import get_inference_plotter

warnings.filterwarnings("ignore", ".*does not have many workers.*")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'predictor', 'defaults.yaml')
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'predictor', 'debug.yaml')
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

def predict(config_path: str, debug: bool = False):
    """Predict a dynamics using a surrogate model."""
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
    check_dict_contains_keys(config, required_keys=[
        'outfolder', 'nexp', 'overwrite', 'output_to_file', 'inference_plotter_type',
        'propagator_train_path', 'propagator_type'
    ])
    config['outpath'] = config.get('outpath', 
                            os.path.join(config['propagator_train_path'], 'predict_runs'))
    if not os.path.isdir(config['propagator_train_path']):
        raise FileNotFoundError(f"Network training directory not found at {config['propagator_train_path']}")

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
        check_dict_contains_keys(config, required_keys=['load_datamodule'])
        dmod_path = config['load_datamodule']
        _log.info("Loading datamodule from: " + dmod_path)
        dmod_ckpt = os.path.join(dmod_path, "datamodule.pth")
        
        # torch.serialization.add_safe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(dmod_ckpt)) # !!! Very Unsafe, only do this if you trust the source of the checkpoint !!!
        dm = torch.load(dmod_ckpt, weights_only=False)
        dm_args = dm.get_args()
        dm_override_args = config.get('datamodule_args', {})
        for key, value in dm_override_args.items():
            if key not in OVERRIDABLE_DMOD_ARGS:
                raise ValueError(f"Cannot override datamodule argument '{key}'. "
                                 f"Allowed keys: {OVERRIDABLE_DMOD_ARGS}")
            _log.info(f"Overriding datamodule argument '{key}' with "
                      f"value: {value}, previous value: {getattr(dm, key, 'N/A')}")
            dm_args[key] = value
            setattr(dm, key, value)
    else:
        check_dict_contains_keys(config, required_keys=[
            'datamodule_type', 'datamodule_args'
        ])
        dm_type = config['datamodule_type']
        dm_args = config['datamodule_args']
        dm_cls = get_datamodule(dm_type)
        dm_args['train_size'] = 0
        dm_args['batch_size'] = 0
        dm_args['predict_size'] = config['inference_args']['predict_steps']
        dm_args['max_frames'] = config['inference_args']['predict_steps']
        dm_args['sequential'] = True
        validate_required_fields(dm_args, get_required_init_args(dm_cls))
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
        analyser.write_data(dm.get_predict_dataset(), label="predict")

    ##################################
    # Setting up the NN
    ##################################
    nn_path = config['propagator_train_path']
    ckpt_path = os.path.join(nn_path, "checkpoints")
    potential_ckpts = [a for a in os.listdir(ckpt_path) if a.endswith(".ckpt")]
    if len(potential_ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
    for name in ['best', 'saved', 'last']:
        if name + ".ckpt" in potential_ckpts:
            nn_ckpt = os.path.join(ckpt_path, name + ".ckpt")
            break
    else:
        _log.warning("No 'best', 'saved', or 'last' checkpoint found. "
                        "Using the first available checkpoint.") 
        nn_ckpt = os.path.join(ckpt_path, potential_ckpts[0])
    _log.info("Loading network from: " + nn_ckpt)
    
    nn_type = config['propagator_type']
    nn_cls = get_propagator(nn_type)
    
    # ------------------------------------------------------------------
    # For backward compatibility with old checkpoints
    # ------------------------------------------------------------------
    hprams = torch.load(nn_ckpt, weights_only=False, map_location='cpu')['hyper_parameters']
    if 'propagator_args' in hprams:
        _log.info("Checkpoint is from an older version. Updating config with checkpoint hyperparameters for compatibility.")
        args = hprams['propagator_args']
        for name in [
            'likelihood', 
            'likelihood_args',
            'encdec_model',
            ]:
            if name in hprams:
                _log.info(f"Overriding argument '{name}' from checkpoint. ")
                args[name] = hprams[name]
        prop = nn_cls.load_from_checkpoint(nn_ckpt, 
                                        strict=False,
                                        datamodule=dm,
                                        args=args,
                                        **metargs)
    # ------------------------------------------------------------------
    else:
        prop = nn_cls.load_from_checkpoint(nn_ckpt, 
                                            datamodule=dm,
                                            **metargs)

    prop.set_inference_settings(
        **config['inference_args']
    )

    ##################################
    # Prediction
    ##################################
    InferencePlotter = get_inference_plotter(config['inference_plotter_type'])
    pred_writer = InferencePlotter(args=config.get('inference_plotter_args', {}), **metargs)

    predictor_args = {}
    predictor_args['default_root_dir'] = run_dir
    predictor_args['accelerator'] = "cpu"
    predictor_args['devices'] = 1
    predictor_args['callbacks'] = [pred_writer]

    predictor = pl.Trainer(**predictor_args)

    predictor.predict(prop, datamodule=dm, return_predictions=False)


def main():
    """Main entry point for the collective encoder testing."""
    args = parse_args()
    predict(args.config, args.debug)

if __name__ == "__main__":
    main()





