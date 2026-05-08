import logging
import os
import yaml

import argparse
import warnings

_log = logging.getLogger(__name__)

import numpy as np

import torch

import pytorch_lightning as pl

from gslibs.utils.common import recursive_update
from gslibs.utils.common import get_required_init_args

from collective_encoder.nets.resolver import get_net
from datamodules.resolver import get_datamodule
from collective_encoder.common.config_check import (
    validate_duplicate_keys, 
)
from collective_encoder.dataanalysers.resolver import get_dataanalyser
from gslibs.utils.filesystem import create_rundir, output_to_file
from collective_encoder.utils import check_dict_contains_keys

warnings.filterwarnings("ignore", ".*does not have many workers.*")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 
                                   'configs', 'dmod', 'defaults.yaml')
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 
                                 'configs', 'dmod', 'debug.yaml')
torch.set_default_dtype(torch.float64)
##################################
# Arguments
##################################
def parse_args():
    desc = "Prepare datamodule for training a collective encoder model based on the provided configuration."
    parser = argparse.ArgumentParser(description=desc)

    # Run Settings
    parser.add_argument('--config', required=True, type=str,
                        help='')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small data and epochs')
    
    args = parser.parse_args()

    return args

def prepare(config_path: str, debug: bool = False):
    """Prepare the datamodule for training a collective encoder model based on the provided configuration."""
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
        'outpath', 'outfolder', 'nexp', 'overwrite', 'output_to_file',
        'datamodule_type', 'datamodule_args',
    ])

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
    # Importing Lightning Modules
    ##################################
    dm_type = config['datamodule_type']
    
    dm_args = config['datamodule_args']
    dm_cls = get_datamodule(dm_type)
    
    dataset_type = dm_args.get('dataset_type', None)
    if dataset_type not in dm_cls._COMPATIBLE_DATASETS:
        raise ValueError(f"Datamodule '{dm_type}' is not compatible with dataset" \
                         f" '{dataset_type}'")

    ##################################
    # Creating Dataset
    ##################################
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
        analyser = analyser_cls(da_args, **metargs)
        analyser.write_data(dm.get_train_dataset(), label="train")
        analyser.write_data(dm.get_val_dataset(), label="val")
        if not dm_args.get('test_whole_dataset', False):
            analyser.write_data(dm.get_test_dataset(), label="test")
    
    ##################################
    # Saving datamodule and config
    ##################################
    torch.save(dm, os.path.join(run_dir, "datamodule.pth"))
    yaml.dump({**dm_args, **metargs}, open(os.path.join(run_dir, "config.yaml"), 'w'))


def main():
    """Main entry point for preparing the datamodule."""
    args = parse_args()
    prepare(args.config, args.debug)

if __name__ == "__main__":
    main()