from typing import Optional

import torch
import vbll
from torch import nn


# Thompson sampling -------------------
class VBLLMLP(nn.Module):
    """
    An MLP model with a VBLL last layer.
    cfg: a config containing model parameters.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        prior_scale: float,
        wishart_scale: float,
        fix_noise_initialization: bool,
        force_prior_scale: bool
    ):
        super(VBLLMLP, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features

        self.params = nn.ModuleDict({
            'in_layer': nn.Linear(in_features, hidden_features),
            'core': nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(num_layers)]),
            'out_layer': vbll.Regression(
                hidden_features,
                out_features,
                regularization_weight=1.,
                prior_scale=prior_scale,
                wishart_scale=wishart_scale,
                parameterization='dense_precision'
            )
        })
        if fix_noise_initialization:
            self.params['out_layer'].noise_logdiag = nn.Parameter(torch.zeros(out_features))
        if force_prior_scale:
            self.params['out_layer'].prior_scale = prior_scale

        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.params['in_layer'](x))
        for layer in self.params['core']:
            x = self.activation(layer(x))
        return self.params['out_layer'](x)

    def sample_posterior_function(self, sample_shape: Optional[torch.Size] = None,):
        if sample_shape is None:
            sampled_params = self.params['out_layer'].W().rsample().to()
        else:
            sampled_params = self.params['out_layer'].W().rsample(sample_shape).to()

        def sampled_parametric_function(x):
            x = self.activation(self.params['in_layer'](x))
            for layer in self.params['core']:
                x = self.activation(layer(x))

            if sample_shape is None:
                return (sampled_params @ x[..., None]).squeeze(-1)

            x_expanded = x.unsqueeze(0).expand(sampled_params.shape[0], -1, -1)
            output = torch.matmul(sampled_params, x_expanded.transpose(-1, -2))
            return output

        return sampled_parametric_function
# Thompson sampling end -------------------