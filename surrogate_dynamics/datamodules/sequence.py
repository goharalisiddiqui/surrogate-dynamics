import warnings
from collective_encoder.datamodules.coordinates import CoordinatesDataModule

warnings.filterwarnings("ignore")

class SequenceDataModule(CoordinatesDataModule):
    '''
    DataModule for sequence prediction tasks. It loads sequences as chunks of subsequent frames from a long sequence. 
    It calculates the correct number of frames for a specific batch size.
    
    Args:
    - input_chunk_length: Number of frames used as input for the model.
    - output_chunk_length: Number of frames used as output for the model.
    - n_seq_per_sample: Number of chunks per sequence.
    
        The total sequence length (one data point) is calculated as:
        sequence_length = input_chunk_length + output_chunk_length * n_seq_per_sample
    '''

    _IDENTIFIER = "SEQUENCE"
    _COMPATIBLE_DATAREADERS = ["XTC_CHUNKS", "XTC_CHUNKS_CG", "XTC_CHUNKS_CG_PP"]
    _COMPATIBLE_DATASETS = ["DISTANCES", "POSITIONS", "GRAPH", "GRAPH_LATENT"]
    _REQUIRED_ARGS = CoordinatesDataModule._REQUIRED_ARGS + [
        "input_chunk_length",
        "output_chunk_length",
    ]
    _OPTIONAL_ARGS = CoordinatesDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "n_seq_per_sample": 1,
    })

    def _initialize_datareader(self):
        chunk_length = self.input_chunk_length + self.output_chunk_length
        sequence_length = (self.input_chunk_length + 
                           self.output_chunk_length * self.n_seq_per_sample)
        self.sequence_length = sequence_length
        self.chunk_length = chunk_length
        self.datareader_args.update({
            "sequence_length": sequence_length
        })
        super()._initialize_datareader()
    
    def get_dataloader(self, data, batch_size, shuffle=False):
        batch_size = batch_size * self.sequence_length
        return super().get_dataloader(data, batch_size, shuffle=False)
