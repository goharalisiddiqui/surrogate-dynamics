import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.distributions.normal import Normal
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2
import pytorch_lightning.loggers as pl_loggers


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from statistics import mean as list_mean

from scipy.stats import multivariate_normal



class AEBase(pl.LightningModule):
    def __init__(self,
                 dim_data : int,
                 dim_latent : int,
                 lr : float = 0.01,
                 l2_reg : float = 1e-7,
                 lr_scheduler : bool = False,
                 outname : str = './untitled/untitled_',
                 plot_every : int = 0,
                 saveplotdata : bool = False,
                 plot_points_limit : int = 20000):
        super().__init__()


        # Model meta info
        self.normIn = False
        self.metaD = False
        self.dim_data = dim_data
        self.dim_latent = dim_latent
        self.plot_points_limit = plot_points_limit


        self.losses = {}
        self.losses["loss"] = []

        self.metaD = False
        self.register_buffer('Mean', torch.zeros(dim_data))
        self.register_buffer('Range', torch.ones(dim_data))

    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        Range[Range == 0.0] = 1.0
        print(f"[{type(self).__name__}] Setting normalization for inputs.")
        self.normIn = True
        self.Mean = Mean
        self.Range = Range

    def normalize(self, x: Variable):
        if not self.normIn:
            return x
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean).div(Range)

    def denormalize(self, x: Variable):
        if not self.normIn:
            return x
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.mul(Range).add(Mean)

    def reparametrize(self, mu, logvar): # Drawing a random sample from the normal distribution mu, logvar
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def reparametrize_multivariate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        dist = torch.distributions.MultivariateNormal(torch.zeros(mu.shape[1]), torch.eye(mu.shape[1]))
        samples = dist.rsample(mu.shape[:-1]).to(mu.device)
        return mu + samples*std



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)
        if self.hparams.lr_scheduler == False:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=3,
                                                       min_lr=1e-10,
                                                       cooldown = 10,
                                                       verbose =True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            }
        }

    def on_train_start(self):
        print("\n\n==================================")
        print(f"Starting training {type(self).__name__} module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lr)
        print("- l2 regularization \t=", self.hparams.l2_reg)
        print("[Hyperparameters]")
        self.print_hparams()
        print("==================================\n\n")

    def print_hparams(self):
        # This prints the hparams in child models
        return

    def training_step(self, train_batch, batch_idx):
        data = self.normalize(train_batch[0].float())
        result, meta = self(train_batch[0].float())
        losses = self.loss(result, data, **meta)

        for key in losses.keys():
            loss_value = losses[key].item() if isinstance(losses[key], torch.Tensor) else losses[key]
            if key not in self.losses.keys():
                self.losses[key] = []
            if batch_idx == 0:
                self.losses[key].append(loss_value)
            else:
                self.losses[key][-1] *= batch_idx
                self.losses[key][-1] += loss_value
                self.losses[key][-1] /= batch_idx + 1

        return losses['loss']

    def on_train_epoch_end(self):
        if self.current_epoch > 0 and \
                self.hparams.plot_every > 0 and \
                    self.current_epoch % self.hparams.plot_every == 0:
            self.plot_test()
        return

    def validation_step(self, val_batch, batch_idx):
        data = self.normalize(val_batch[0].float())
        result, meta = self(val_batch[0].float())
        losses = self.loss(result, data, **meta)

        for key in losses.keys():
            loss_value = losses[key].item() if isinstance(losses[key], torch.Tensor) else losses[key]
            mkey = "val_" + key
            if mkey not in self.losses.keys():
                self.losses[mkey] = []
            if batch_idx == 0:
                self.losses[mkey].append(loss_value)
            else:
                self.losses[mkey][-1] *= batch_idx
                self.losses[mkey][-1] += loss_value
                self.losses[mkey][-1] /= batch_idx + 1

        for key in self.losses.keys():
            if len(self.losses[key]) > 0:
                self.log(key, self.losses[key][-1], prog_bar=True)
        return losses['loss']

    def plot_test(self):
        dataloader = self.trainer.datamodule.test_dataloader()
        data_batch = [t.to(self.device) for t in next(iter(dataloader))]
        self.test_step(data_batch, 0)

    def test_step(self, test_batch, batch_idx):
        self.plot_training()
        data = self.normalize(test_batch[0].float())
        if len(test_batch) > 1:
            labels = test_batch[1].float()
            labels = labels.cpu().detach().numpy()
        else:
            labels = None
        latents = self.get_latent(test_batch[0].float())
        if isinstance(latents, tuple):
            latent = latents[0]
        else:
            latent = latents
        self.print_labels_latent_correlations(latent, labels)
        if self.hparams.saveplotdata:
            self.export_latent(latents, labels)
        self.plot_latent(latents, labels=labels)
        self.plot_extra(data, labels, latents)

    def plot_extra(self, data_x, data_y, latents):
        # This implements any extra printing or plotting in child class
       return

    def get_latent(self, data_x):
        raise NotImplementedError("This function must be implemented in child class")

    def print_fve(self, datamodule):
        dl = datamodule.test_dataloader()
        flag = self.training
        self.training = False
        with torch.no_grad():
            data = next(iter(dl))[0].float()
            output, _ = self(data)
            target = self.normalize(data)
            sub = torch.sub(target, output)
            ss_err = torch.sum(torch.pow(sub, 2), dim=0)
            meann = torch.mean(target, dim=0, keepdim=True)
            sub_meann = torch.sub(target, meann)
            ss_tot = torch.sum(torch.pow(sub_meann, 2), dim=0)
            fve = 1 - torch.div(ss_err, ss_tot)
            fve_mean = torch.mean(fve).detach().cpu().numpy() # This calculates FVE for each input dim and mean it

            ss_err = torch.sum(torch.pow(sub, 2))
            ss_tot = torch.sum(torch.pow(sub_meann, 2))
            fve_sum = 1 - torch.div(ss_err, ss_tot).detach().cpu().numpy() # This calculates one FVE by taking inner product of vectors instead of square
        print("\n\n=======================================")
        print("Fraction of Variation Explined (FVE)")
        print("=======================================")
        # print("FVE_mean = ", fve_mean)
        print("FVE = ", fve_sum)
        print("=======================================\n\n")
        self.training = flag
        return fve_mean

    def print_labels_latent_correlations(self, latent, labels = None):
        # Calculates and prints correlation between labels+latent_space
        if labels is None:
            data_df = pd.DataFrame(latent, columns=["Latent Dimension %d"%i for i in range(latent.shape[1])])
        else:
            data_df = pd.DataFrame(np.concatenate((latent, labels), axis=1), columns=["Latent Dimension %d"%i for i in range(latent.shape[1])] + self.trainer.datamodule.label_list)
        print("\n\n=======================================")
        print("Correlation of latent space and labels (if present)")
        print("=======================================")
        print(data_df.corr())
        print("=======================================\n\n")

        fig = plt.figure(figsize=(19, 15))
        plt.matshow(data_df.corr().abs(), fignum=fig.number)
        plt.xticks(range(data_df.columns.shape[0]), data_df.columns.tolist(), fontsize=14, rotation=45)
        plt.yticks(range(data_df.columns.shape[0]), data_df.columns.tolist(), fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        self.log_tbimage("Correlation", fig)

    def plot_training(self):
        if len(self.losses["loss"]) == 0:
            return
        if len(self.losses.keys()) == 0 or "loss" not in self.losses.keys():
            raise Exception("No losses to plot.")
        non_val_losses = {key: self.losses[key] for key in self.losses.keys() if "val_" not in key}
        n_plots = len(non_val_losses.keys())
        fig, ax = plt.subplots(1, n_plots, squeeze=True, figsize=(6 * n_plots, 6))
        ax[0].set_title("Network Loss minimization")
        if all(np.asarray(non_val_losses["loss"]) > 0.0):
            ax[0].set_yscale("log")
        lns = ax[0].plot(
            np.asarray(non_val_losses["loss"]),
            "o-",
            label="Test (left)",
            c="tab:red",
            alpha=0.3,
        )
        if "val_loss" in self.losses.keys():
            ax2 = ax[0].twinx()
            if all(np.asarray(self.losses["val_loss"]) > 0.0):
                ax2.set_yscale("log")
            lns += ax2.plot(
                np.asarray(self.losses["val_loss"]),
                ".-",
                label="Validation (right)",
                c="tab:red",
            )
        ax[0].set_xlabel("Epoch")
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc=0)
        # ax[0].set_ylabel("Loss")

        colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

        for ind, key in enumerate([key for key in non_val_losses.keys() if key != "loss"]):
            i = ind + 1
            ax[i].set_title(f"{key} minimization")
            if all(np.asarray(non_val_losses[key]) > 0.0):
                ax[i].set_yscale("log")
            lns = ax[i].plot(
                np.asarray(non_val_losses[key]),
                "o-",
                label="Test (left)",
                c=colors[i % len(colors)],
                alpha=0.3,
            )
            if "val_" + key in self.losses.keys():
                ax2 = ax[i].twinx()
                if all(np.asarray(non_val_losses[key]) > 0.0):
                    ax2.set_yscale("log")
                lns += ax2.plot(
                    np.asarray(self.losses["val_" + key]),
                    ".-",
                    label="Validation (right)",
                    c=colors[i % len(colors)],
                )
            ax[i].set_xlabel("Epoch")
            labs = [l.get_label() for l in lns]
            ax[i].legend(lns, labs, loc=0)
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{self.current_epoch}_training.png", dpi=150)
        logger = self.logger
        if isinstance(logger, pl_loggers.TensorBoardLogger):
            logger = logger.experiment
            logger.add_figure(f"Training", fig, self.current_epoch)
        plt.close()

    @torch.no_grad()
    def export_serial_model(self, model_path = None):
        print(f"[Exporting the serialized {type(self).__name__} model]")

        fake_loader = self.trainer.datamodule.test_dataloader()
        fake_input = next(iter(fake_loader))[0].float()

        if model_path == None:
            model_path = '.'
        if not os.path.isdir(model_path):
                os.makedirs(model_path)

        self.metaD = True
        torch.jit.save(self.to_torchscript(method='trace', example_inputs=fake_input, strict=False), f"{model_path}/encoder.pt")
        self.metaD = False
        print(f"[{type(self).__name__} model serialized at: {model_path}/encoder.pt]")


    @torch.no_grad()
    def export_latent(self, latents, labels = None):
        filename_stem = f"{self.hparams.outname}{self.current_epoch}_"

        if isinstance(latents, tuple):
            latent_names = self.get_latent_names()
            assert len(latents) == len(latent_names)
            for i in range(len(latents)):
                np.save(filename_stem + f"{latent_names[i]}.npy", latents[i])
        if labels is not None:
            np.save(filename_stem + f"labels.npy", labels)

    @torch.no_grad()
    def plot_latent(self, latents, labels = None, plot_func : callable = None, tag = None):
        if labels is not None:
            label_list = self.trainer.datamodule.label_list
        latent_len = len(latents) if isinstance(latents, tuple) else 1
        plot_len = latents[0].shape[0] if isinstance(latents, tuple) else latents.shape[0]

        if plot_len > self.plot_points_limit:
            print(f"[WARNING] Limiting plot to {self.plot_points_limit} points")
            index = np.random.choice(plot_len, self.plot_points_limit, replace=False)
            if latent_len > 1:
                latents = tuple([latent[index] for latent in latents])
            else:
                latents = latents[index]
            if labels is not None:
                labels = labels[index]


        n_fig = 10
        k = 0
        n_hidden = self.dim_latent
        n_cols = labels.shape[1] if labels is not None else 1
        while (n_hidden > 0):
            n_rows = n_hidden if n_hidden > 2 else 1
            if n_rows > n_fig:
                n_rows = n_fig
            fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(6 * n_cols, 6 * n_rows))

            for i in range(0, axes.shape[0]):
                for j in range(n_cols):
                    self.plot_latent_axis(fig, axes[i][j], latents, labels[:,j] if labels is not None else None, i, j, label_list[j] if labels is not None else None, plot_func)
            n_hidden -= n_rows
            if n_hidden == 1:
                n_hidden = 0
            plt.tight_layout()
            if tag is None:
                tag = "latent_space"
            if k == 0 and n_hidden == 0:
                figoname = f"{self.hparams.outname}{self.current_epoch}_{tag}.png"

            else:
                figoname = f"{self.hparams.outname}{self.current_epoch}_{tag}_{k+1}.png"
            fig.savefig(figoname, dpi=150)
            self.log_tbimage(f"{tag}" if k == 0 and n_hidden == 0 else f"{tag}-{k+1}", fig)
            plt.close()
            k += 1

    @torch.no_grad()
    def plot_latent_axis(self, fig, ax, latents, train_y, i, j, label, plot_func):
        ax.set_title(f"{type(self).__name__} Latent-space-"+str(i))

        if train_y is not None:
            cm = plt.get_cmap('jet')
            cNorm = matplotlib.colors.Normalize(vmin=min(train_y), vmax=max(train_y))
            scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        else:
            scalarMap = None

        latent = latents[0] if isinstance(latents, tuple) else latents

        yaxis = (i+1) if (i+1) < latent.shape[1] else 0

        if plot_func is not None:
            plot_func(fig, ax, latents, train_y, i, yaxis, label, scalarMap)
        else:
            ax.scatter(latent[:, i], latent[:, yaxis], c=scalarMap.to_rgba(train_y) if train_y is not None else None, label="Whole dataset", alpha=0.3)
        ax.set_xlabel(f"Latent Dimension {i}")
        ax.set_ylabel(f"Latent Dimension {yaxis}")

        if train_y is not None:
            scalarMap.set_array(train_y)
            cb = fig.colorbar(scalarMap, ax=ax)
            cb.set_label(label if label else "Label-"+str(j))

    @torch.no_grad()
    def log_tbimage(self, tag, image, step = None):

        logger = self.logger
        if isinstance(logger, pl_loggers.TensorBoardLogger):
            logger = logger.experiment
            logger.add_figure(tag, image, step if step is not None else self.current_epoch)

