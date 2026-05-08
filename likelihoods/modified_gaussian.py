import torch
from darts.utils.likelihood_models import GaussianLikelihood

class ModifiedGaussianLikelihood(GaussianLikelihood):
    def sample(self, 
               model_output: torch.Tensor, 
               temperature: float = 1.0, 
               fixed_sigma: bool = False) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        if fixed_sigma:
            return torch.normal(mu, temperature)
        sigma = sigma * temperature
        
        return torch.normal(mu, sigma)

