import os
import numpy as np
import torch
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter


class Ala2Writer(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        assert len(predictions) == 1, "Expecting a single batch of predictions"
        # for k, v in predictions[0].items():
        #     print(f"key: {k}, value shape: {v.shape}")
        predictions = predictions[0]  # Always expecting a single batch

        decoded_cos_true = predictions['dihedral_cos_true'].cpu().numpy()
        decoded_sin_true = predictions['dihedral_sin_true'].cpu().numpy()

        decoded_cos_pred = predictions['dihedral_cos_pred'].cpu().numpy()
        decoded_sin_pred = predictions['dihedral_sin_pred'].cpu().numpy()

        # _, __ , torsion_index = self.get_dataset().get_label_indices()
        # print(f"Torsion indices: {torsion_index}")
        # exit()
        idx_phi = 6 # [1,3,4,5]
        idx_psi = 10 # [3,4,6,8]

        phi = np.arctan2(decoded_sin_true[:,idx_phi], decoded_cos_true[:,idx_phi])
        psi = np.arctan2(decoded_sin_true[:,idx_psi], decoded_cos_true[:,idx_psi])

        phi_pred = np.arctan2(decoded_sin_pred[:,idx_phi], decoded_cos_pred[:,idx_phi])
        psi_pred = np.arctan2(decoded_sin_pred[:,idx_psi], decoded_cos_pred[:,idx_psi])

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.scatter(phi, psi, s=1)
        plt.xlabel("Phi")
        plt.ylabel("Psi")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.subplot(1,2,2)
        plt.title("Predicted")
        plt.scatter(phi_pred, psi_pred, s=1)
        plt.xlabel("Phi")
        plt.ylabel("Psi")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.savefig(self.output_dir + "/ramachandran.png", dpi=300)
        plt.close()
        print(f"\n[{type(self).__name__}]: Saved Ramachandran plot to {self.output_dir}ramachandran.png")