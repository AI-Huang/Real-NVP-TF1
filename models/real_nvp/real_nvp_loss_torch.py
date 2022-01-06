#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-17-21 21:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

# Ported from https://github.com/chrischute/real-nvp


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        q (int or float): Number of discrete values (quantization levels) in each input dimension.
            E.g., `q` is 256 for natural images.

    See Also:
        Equation(3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, q=256):
        super(RealNVPLoss, self).__init__()
        self.q = q

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1)

        K = np.prod(z.size()[1:])
        prior_ll -= np.log(self.q) * K

        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll

# Adjusted Real NVP loss for Lossy Image Compression


class CompressionLoss(nn.Module):
    """Get the NLL loss for a RealNVPCompression model.

    Args:
        q (int or float): Number of discrete values in each input dimension.
            E.g., `q` is 256 for natural images.
        lambda_ (int or float): Penalization coefficient on the deviation between $\hat x$ and $x$, or $\tilde x$ and $x$.
            E.g., lambda_=500.

    Reference:
        Equation(7) and (10) in Lossy Image Compression with Norfmalizing Flows: https://arxiv.org/abs/2008.10486
    """

    def __init__(self, q=256, lambda_=500, dims=None):
        super(CompressionLoss, self).__init__()
        assert dims is not None
        self.dims = dims
        self.q = q
        self.lambda_ = lambda_
        self.d_hat = None
        self.d_tilde = None
        self.prior = distributions.MultivariateNormal(
            torch.zeros(dims), torch.eye(dims))

    def forward(self, z, sldj, x=None, x_hat=None, x_tilde=None):
        # prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1)

        prior_ll = self.prior.log_prob(z)

        # dim = np.prod(z.size()[1:])
        prior_ll -= np.log(self.q) * self.dim

        ll = prior_ll + sldj
        nll = -ll.mean()

        loss = nll
        if not x is None and not x_hat is None:
            self.d_hat = self.lambda_ * (F.mse_loss(x, x_hat))
            loss += self.d_hat
        if not x is None and not x_tilde is None:
            self.d_tilde = self.lambda_ * (F.mse_loss(x, x_tilde))
            loss += self.d_tilde

        return loss, nll


def main():
    pass


if __name__ == "__main__":
    main()
