import numpy as np
import torch
from dataclasses import dataclass
from vbll.utils.distributions import Normal, DenseNormal, LowRankNormal, DenseNormalPrec, get_parameterization
from collections.abc import Callable
import torch.nn as nn


def gaussian_kl(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(mse_term + trace_term + logdet_term) # currently exclude constant


def gamma_kl(cov_dist, prior_dist):
    kl = torch.distributions.kl.kl_divergence(cov_dist, prior_dist)
    return (kl).sum(-1)


def expected_gaussian_kl(p, q_scale, cov_dist):
    cov_factor = cov_dist.concentration / cov_dist.rate

    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1)/ q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    combined_mse_term = (cov_factor * mse_term).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(combined_mse_term + trace_term + logdet_term) # currently exclude constant


@dataclass
class VBLLReturn():
    predictive: Normal | DenseNormal | torch.distributions.studentT.StudentT
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None


class Regression(nn.Module):
    """
    Variational Bayesian Linear Regression

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}
    prior_scale : float
        Scale of prior covariance matrix
    wishart_scale : float
        Scale of Wishart prior on noise covariance
    init_noise_logdiag : str
        Initialization of log diagonal of covariance matrix. Currently supports {'random', 'zeros'}
    dof : float
        Degrees of freedom of Wishart prior on noise covariance
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=1.,
                 wishart_scale=1e-2,
                 cov_rank=None,
                 init_noise_logdiag='random',
                 dof=1.):
        super(Regression, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1. / in_features) 

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)
        if init_noise_logdiag == 'zeros':
            self.noise_logdiag = nn.Parameter(torch.zeros(out_features))
        else:
            self.noise_logdiag = nn.Parameter(torch.randn(out_features) * (np.log(wishart_scale)))

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))

        # W and noise cache
        self.W_cache = None
        self.noise_cache = None
        
        if parameterization == 'diagonal':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
        elif parameterization == 'dense':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features)/in_features)
        elif parameterization == 'dense_precision':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) + 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features)*0.0)
        elif parameterization == 'lowrank':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, cov_rank)/in_features)

    def W(self):
        if self.training:
            self.W_cache = None
        if not self.training and self.W_cache is not None:
            return self.W_cache
        
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)
        
        if not self.training:
            self.W_cache = cov
        
        return cov

    def noise(self):
        if self.training:
            self.noise_cache = None
        if not self.training and self.noise_cache is not None:
            return self.noise_cache
        
        normal = Normal(self.noise_mean, torch.exp(self.noise_logdiag))

        if not self.training:
            self.noise_cache = normal
        
        return normal

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):

        def loss_fn(y, recursive_update = False):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[...,None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5*((W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])) * noise.trace_precision)

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            if recursive_update:
                assert(self.W_dist == DenseNormalPrec)
                with torch.no_grad():
                    noise_cov = torch.exp(self.noise_logdiag) # out_dim * 1
                    chol, new_mean = self.W().recursive_update(x, y, noise_cov)

                    # update in place
                    self.W_logdiag.data = torch.log(torch.diagonal(chol, dim1 = -2, dim2=-1))
                    self.W_offdiag.data = torch.tril(chol, diagonal = -1)
                    self.W_mean.data = new_mean
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1) # sum over output dims            
            return -logprob.mean(0) # mean over batch dim

        return loss_fn
    
    def create_thompson_heads(self, n: int=1):
        """Creates n Thompson samples from the posterior distribution.

        Args:
            n: int, optional (default=1) number of samples to draw from the posterior distribution.
        
        Returns:
            heads: List of n tensors of shape (out_features, in_features) representing the weights of the extracted layer"""
        is_training = self.training
        self.eval()
        heads = []
        for _ in range(n):
            W = self.W()
            heads.append(W.sample())
        if is_training:
            self.train()
        return heads
    
    def create_mean_thompson_head(self):
        """
        Create one Thompson sample from the posterior distribution. Namely it gives back the mean of the approx of the posterior distribution.

        :return: tensor of shape (out_features, in_features) representing the weight of the extracted layer
        """
        return [self.W_mean]


class tRegression(nn.Module):
    """
    Variational Bayesian Linear Student-t Regression
    
    This version of the VBLL regression layer also infers noise covariance.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank'}
    prior_scale : float
        Scale of prior covariance matrix
    wishart_scale : float
        Scale of Wishart prior on noise covariance
    dof : float
        Degrees of freedom of Wishart prior on noise covariance
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=100.,
                 wishart_scale=100.,
                 cov_rank=None,
                 dof=1.):
        super(tRegression, self).__init__()

        self.wishart_scale = wishart_scale
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_dof = dof
        self.prior_rate = 1./wishart_scale
        self.prior_scale = prior_scale * (2. / in_features) # kaiming init

        # variational posterior over noise params
        self.noise_log_dof = nn.Parameter(torch.ones(out_features) * np.log(self.prior_dof))
        self.noise_log_rate = nn.Parameter(torch.ones(out_features) * np.log(self.prior_rate))

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))

        self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features))
        if parameterization == 'dense':
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features))
        elif parameterization == 'lowrank':
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, cov_rank))
        elif parameterization == 'dense_precision':
            raise NotImplementedError()
            

    @property
    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif self.W_dist == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    @property
    def noise(self):
      noise_dof = torch.exp(self.noise_log_dof)
      noise_rate = torch.exp(self.noise_log_rate)
      return torch.distributions.gamma.Gamma(noise_dof, noise_rate)

    @property
    def noise_prior(self):
      return torch.distributions.gamma.Gamma(self.prior_dof, self.prior_rate)

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    def predictive(self, x):
        dof = 2 * self.noise.concentration
        Wx = (self.W @ x[..., None]).squeeze(-1)
        mean = Wx.mean
        pred_cov = (Wx.variance + 1) * self.noise.rate / self.noise.concentration
        return torch.distributions.studentT.StudentT(dof, mean, torch.sqrt(pred_cov))

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            cov_factor = self.noise.concentration / self.noise.rate
            pred_err = (y - (self.W.mean @ x[...,None]).squeeze(-1)) ** 2
            pred_likelihood = (cov_factor * pred_err).sum(-1)

            logdet_term = (torch.digamma(self.noise.concentration) - torch.log(self.noise.rate)).sum(-1)
            trace_term = (self.W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])).sum(-1)

            kl_term = expected_gaussian_kl(self.W, self.prior_scale, self.noise)
            kl_term += gamma_kl(self.noise, self.noise_prior)

            total_elbo = -0.5 * torch.mean(pred_likelihood + trace_term - logdet_term)
            total_elbo -= self.regularization_weight * kl_term
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1) # sum over output dims
            return -logprob.mean(0) # mean over batch dim

        return loss_fn
