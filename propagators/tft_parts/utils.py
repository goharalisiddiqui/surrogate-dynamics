import torch
from torch import nn as nn
import torch.nn.functional as F

class MonteCarloDropout(nn.Dropout):
    """
    Defines Monte Carlo dropout Module as defined
    in the paper https://arxiv.org/pdf/1506.02142.pdf.
    In summary, This technique uses the regular dropout
    which can be interpreted as a Bayesian approximation of
    a well-known probabilistic model: the Gaussian process.
    We can treat the many different networks
    (with different neurons dropped out) as Monte Carlo samples
    from the space of all available models. This provides mathematical
    grounds to reason about the model’s uncertainty and, as it turns out,
    often improves its performance.
    """

    # mc dropout is deactivated at init; see `MonteCarloDropout.mc_dropout_enabled` for more info
    _mc_dropout_enabled = False

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: we could use the following line in case a different rate
        # is used for inference:
        # return F.dropout(input, self.applied_rate, True, self.inplace)
        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)

    @property
    def mc_dropout_enabled(self) -> bool:
        # mc dropout is only activated on `PLForecastingModule.on_predict_start()`
        # otherwise, it is activated based on the `model.training` flag.
        return self._mc_dropout_enabled or self.training