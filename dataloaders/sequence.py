import warnings
import random

import numpy as np

import torch
from collective_encoder.dataloaders.default import DefaultDatamodule

import torch

warnings.filterwarnings("ignore")

if True: # for reproducibility
    SEED = 53
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

class SequenceDatamodule(DefaultDatamodule):
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 num_chunks_per_sequence: int = 1,
                 **kwargs
                 ):

        chunk_length = input_chunk_length + output_chunk_length
        sequence_length = input_chunk_length + output_chunk_length * num_chunks_per_sequence
        datareader_args = kwargs.pop("datareader_args", {})
        datareader_args.update({
            "sequence_length": sequence_length,
        })
        datareader_type = kwargs.pop("datareader_type", "XTC_CHUNKS")
        if datareader_type not in self._allowed_datareaders():
            raise ValueError(f"Datareader type {datareader_type} not supported for SequenceDatamodule. "
                             f"Allowed types: {self._allowed_datareaders()}")
        kwargs.update({
            'datareader_args': datareader_args,
            'datareader_type': datareader_type,
        })
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.chunk_length = chunk_length
    
    def _allowed_datareaders(self):
        return ["XTC_CHUNKS", "XTC_CHUNKS_CG"]

    def get_dataloader(self, indices, batch_size):
        batch_size = batch_size * self.sequence_length
        return super().get_dataloader(indices, batch_size)
