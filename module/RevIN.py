import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mask=None, mode: str = 'norm'):
        if mode == "norm":
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == "denorm":
            x = self._denormalize(x)
        elif mode == "denorm_mu_sigma":
            x = self._denormalize_mu_sigma(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        # self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        if mask is None:
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        else:
            count = mask.sum(dim=dim2reduce, keepdim=True).clamp_min(1.0)
            mean = x.sum(dim=dim2reduce, keepdim=True) / count

            mean_sq = (x ** 2).sum(dim=dim2reduce, keepdim=True) / count
            var = mean_sq - mean ** 2

        self.mean = mean.detach()
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x, mask):
        if mask is None:
            mask = torch.ones_like(x)
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        x = x * mask
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def _denormalize_mu_sigma(self, x):
        mu, logvar = x
        if self.affine:
            mu = mu - self.affine_bias
            mu = mu / (self.affine_weight + self.eps * self.eps)
            logvar = logvar - 2 * torch.log(self.affine_weight + self.eps * self.eps)
        mu = mu * self.stdev
        mu = mu + self.mean
        logvar = logvar + 2 * torch.log(self.stdev)
        x = (mu, logvar)
        return x

