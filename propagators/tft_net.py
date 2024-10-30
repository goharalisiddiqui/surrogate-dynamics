import argparse
from darts.models import TFTModel


def tft_args():
    desc = "TFT model Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--seqlength', dest="input_chunk_length", default=5, type=int, help='Number of time steps in the past to take as a model input')
    parser.add_argument('--predlength', dest="output_chunk_length", default=5, type=int, help='Number of time steps to predict at once')
    # parser.add_argument('--nhidden', dest="hidden_size", default=[96], nargs='+', help='Hidden state size of the TFT')
    parser.add_argument('--nhidden', dest="hidden_size", default=96, type=int, help='Hidden state size of the TFT')
    parser.add_argument('--nlstm', dest="lstm_layers", default=2, type=int, help='Number of layers for the LSTM Encoder and Decoder')
    parser.add_argument('--nattention', dest="num_attention_heads", default=1, type=int, help='Number of attention heads')
    parser.add_argument('--dropout', dest="dropout", default=0.02, type=float, help='Fraction of neurons affected by dropout.')
    parser.add_argument('--batch_size', dest="batch_size", default=100, type=int, help='Number of time series (input and output sequences) used in each training pass.')

    args, _ = parser.parse_known_args()

    return args

TFT_args = tft_args
