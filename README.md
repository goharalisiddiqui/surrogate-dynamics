# Surrogate Dynamics

A machine learning framework for learning and predicting molecular dynamics trajectories using neural networks as surrogate models.

## Core Idea

**Surrogate Dynamics** trains neural networks to learn and predict the temporal evolution of molecular systems from molecular dynamics (MD) simulation data. Rather than running computationally expensive simulations, the framework enables fast, differentiable predictions of molecular trajectories.

The approach combines:
- **Encoding**: A graph neural network encoder compresses 3D atomic structures and their properties into a compact latent representation
- **Temporal Prediction**: A Temporal Fusion Transformer (TFT) predicts future latent representations as a time series
- **Decoding**: The predicted latent codes are decoded back into atomic coordinates and properties

This enables rapid inference of molecular dynamics without the computational overhead of traditional MD simulations, making it suitable for optimization, sampling, and exploratory tasks.

## Project Structure

```
surrogate-dynamics/
├── trainer.py                      # Main training orchestrator
├── predictor.py                    # Inference/prediction script
├── utils_sd.py                     # Utility functions and helpers
├── requirements.txt                # Python dependencies
│
├── config_train_*.yaml             # Training configuration files
│   ├── config_train_encoder.yaml
│   ├── config_train_propagator.yaml
│   ├── config_train_e2e.yaml       # End-to-end training
│   └── config_predict.yaml
│
├── run_*.sh                        # Bash scripts for training/prediction
│   ├── run_train_encoder.sh
│   ├── run_train_propagator.sh
│   ├── run_train_e2e.sh
│   └── run_predict.sh
│
├── propagators/                    # Temporal prediction models
│   ├── tft.py                      # Temporal Fusion Transformer
│   ├── bge_tft.py                  # Bond Graph Encoder + TFT
│   ├── bge_tft_v2.py               # Bond Graph Encoder v2 + TFT
│   └── tft_model/                  # TFT model implementation
│
├── embeddings/                     # Encoder/decoder models
│   ├── resolver.py                 # Model resolver for encoder selection
│   └── flatemb.py                  # Flat embedding layer
│
├── dataloaders/                    # Data loading and preprocessing
│   ├── sequence.py                 # Sequence datamodule for time series
│   └── xtc_latent.py               # XTC trajectory loading in latent space
│
├── likelihoods/                    # Loss functions and likelihood models
│   ├── resolver.py                 # Likelihood function resolver
│   ├── modified_gaussian.py        # Modified Gaussian likelihood
│   └── modified_quantile.py        # Modified Quantile Regression likelihood
│
├── data_generation/                # Data generation scripts and samples
│   └── ala2/                       # Example MD trajectories (alanine dipeptide)
│
├── plotters/                       # Visualization and analysis tools
│   └── ala2.py                     # ALA2 specific plotting functions
│
├── short_scripts/                  # Utility and debugging scripts
│
├── collective_encoder/             # External submodule for encoder/decoder
│   └── (Graph neural network encoder-decoder for molecular structures)
│
└── inputs/                         # Configuration and input files
```

## Key Components and Classes

### `trainer.py`
Main training orchestrator that:
- Loads configuration from YAML files
- Initializes encoder, propagator, and data modules
- Manages the training loop with PyTorch Lightning
- Supports logging to Weights & Biases (WandB)
- Handles model checkpointing and early stopping
- Supports debug mode for quick testing

**Usage:**
```bash
python trainer.py --config config_train_e2e.yaml
python trainer.py --config config_train_e2e.yaml --debug  # Debug mode
```

### `predictor.py`
Inference script that:
- Loads trained encoder and propagator models
- Performs trajectory prediction on molecular systems
- Generates multiple prediction steps
- Writes predictions to output files
- Supports visualization of predictions

**Usage:**
```bash
python predictor.py --config config_predict.yaml
python predictor.py --config config_predict.yaml --debug
```

### `propagators/tft.py`

**Class: `PropagatorTFT`** (PyTorch Lightning Module)
- Implements a Temporal Fusion Transformer for time series forecasting
- Takes encoder-decoder model as input to transform sequences
- Supports multiple likelihood functions (Gaussian, Quantile Regression, etc.)
- Configurable learning rate scheduling
- Outputs predictions with uncertainty quantification

**Key Arguments:**
- `input_chunk_length`: Number of timesteps for input sequences
- `output_chunk_length`: Number of timesteps to predict
- `hidden_dim`: Hidden dimension size (typically 96-256)
- `likelihood`: Loss function type (Gaussian, Quantile, etc.)
- `lr`: Learning rate

### `propagators/bge_tft.py` and `bge_tft_v2.py`

**Class: `BondGraphEncoderTFT`**
- Enhanced propagator combining Bond Graph Encoder with TFT
- Models molecular systems as graph structures with bond information
- Enforces chemical constraints through bond graph representations
- Improved version 2 with additional refinements

### `dataloaders/sequence.py`

**Class: `SequenceDatamodule`** (PyTorch Lightning DataModule)
- Extends collective encoder's datamodule for sequence-based learning
- Handles chunking of MD trajectories into input/output sequences
- Supports multiple timesteps per sample
- Configurable batch sizes and data loading

**Key Parameters:**
- `input_chunk_length`: Past timesteps to use as input
- `output_chunk_length`: Future timesteps to predict
- `num_chunks_per_sequence`: Number of consecutive predictions per trajectory

### `dataloaders/xtc_latent.py`

**Class: `XtcTrainer`**
- Data loader for XTC trajectory files (GROMACS format)
- Handles trajectory reading and preprocessing
- Converts atomic coordinates to latent space representations
- Supports residue selection and alignment

### `embeddings/resolver.py`

**Function: `get_encdec()`**
- Factory function to instantiate encoder-decoder models
- Currently supports Bond Graph Encoder (BGE)
- Extensible architecture for adding new encoders

### `likelihoods/resolver.py`

**Class: `LikelihoodResolver`**
- Factory for different loss functions and likelihood models
- Supports Gaussian, Laplace, and Quantile Regression
- Handles uncertainty quantification in predictions

## Configuration Files

Configuration is specified via YAML files:

### `config_train_e2e.yaml` (End-to-End Training)
- Trains encoder and propagator jointly
- Configures loss weights for reconstruction and prediction
- Specifies data parameters and training hyperparameters

**Key Parameters:**
- `nepochs`: Number of training epochs
- `lrate`: Learning rate
- `loss_prop_weight`: Weight for propagation loss
- `loss_rec_weight`: Weight for reconstruction loss
- `loss_e2e_weight`: Weight for end-to-end loss
- `batch_size`: Training batch size
- `input_chunk_length`: Input sequence length
- `output_chunk_length`: Output prediction length

### `config_predict.yaml`
- Configuration for running predictions on new trajectories
- Specifies model checkpoints to load
- Configures prediction parameters

## Data Format

### Input Data
- **XTC Files**: GROMACS trajectory files containing atomic coordinates
- **TPR Files**: GROMACS topology files with system structure
- **Selection**: MDAnalysis-style selection strings (e.g., "resname ALA and not element H")

### Output Data
- Predicted molecular trajectories
- Uncertainty estimates (if applicable)
- Visualization plots and analysis files

## Example Dataset: Alanine Dipeptide (ALA2)

The project includes example data for alanine dipeptide (ALA2) trajectories:
```
data_generation/ala2/300K/
├── initialization/
│   └── tpr_initial.tpr          # System topology
└── ala2_100ns/
    └── md.xtc                    # 100 ns MD trajectory
```

## Training Workflow

### 1. Train Encoder (Optional)
If starting from scratch, first train an encoder:
```bash
python trainer.py --config config_train_encoder.yaml
```

### 2. Train Propagator
Train the temporal prediction model:
```bash
python trainer.py --config config_train_propagator.yaml
```

### 3. End-to-End Training
Or train encoder and propagator jointly:
```bash
python trainer.py --config config_train_e2e.yaml
```

## Prediction Workflow

1. Load trained models (encoder + propagator)
2. Run prediction on test trajectories:
```bash
python predictor.py --config config_predict.yaml
```
3. Analyze and visualize predictions using notebooks in `plotters/`

## Key Dependencies

- **PyTorch & PyTorch Lightning**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **MDAnalysis**: MD trajectory analysis
- **ASE**: Atomic simulation environment
- **YAML**: Configuration management
- **Weights & Biases**: Experiment tracking (optional)

See `requirements.txt` for complete list.

## Example Usage

### Training
```bash
# End-to-end training on ALA2 data
python trainer.py --config config_train_e2e.yaml

# Debug mode (quick test with small data)
python trainer.py --config config_train_e2e.yaml --debug
```

### Prediction
```bash
# Run prediction on new trajectories
python predictor.py --config config_predict.yaml

# Debug mode
python predictor.py --config config_predict.yaml --debug
```

## Output Structure

Training and prediction outputs are organized as:
```
train_runs/run_test_1/
├── checkpoints/
│   ├── best.ckpt               # Best model checkpoint
│   └── last.ckpt               # Last epoch checkpoint
├── logs/
│   └── training.log            # Training logs
└── [model_name]_predictions/   # Prediction outputs
```

## Development and Debugging

- Use `--debug` flag for quick testing with minimal data and epochs
- Check `short_scripts/` for utility functions and debugging tools
- Jupyter notebooks in `plotters/` for visualization and analysis
- WandB integration for real-time training monitoring

## References

The project builds on:
- Temporal Fusion Transformer (Lim et al., 2021)
- Graph Neural Networks for molecular systems
- PyTorch Lightning for efficient training

## License

N/A

## Contact

For questions or issues, contact the project maintainers.
