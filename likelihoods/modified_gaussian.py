from darts.utils.likelihood_models import GaussianLikelihood

import torch


class ModifiedGaussianLikelihood(GaussianLikelihood):
    def sample(self, model_output: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        sigma = sigma * temperature
        return torch.normal(mu, sigma)

