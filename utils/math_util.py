#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-23-21 19:40
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import numpy as np
import torch


def quantize(z, delta=1.0, device=None):
    """Quantize the input z

    Args:
        z (torch.Tensor): the input.
    """
    # z = preprocess(z) # Not necessary any more
    device = z.device if device == None else device
    # Quantize using uniform ramdom number
    u = (torch.rand(z.shape)-0.5) * delta
    u = u.to(device)
    z_hat = torch.round(z+u) - u

    return z_hat


def psnr(mse, max_i=255):
    """PSNR

    Args:
        mse (float):
    Return:
        r (float): entropy of x.
    """
    r = 10 * np.log(max_i**2/mse) / np.log(10)  # to dB

    return r


def entropy(x, norm=True, base="binary"):
    """Computer entropy for a 8-bit quantilized image or image batch x.

    Args:
        x (torch.Tensor):
        norm (bool): default True, whether the x should be normalized into [0, 255].
        base (str): default "binary", "nature", "binary", "decimal".
    Return:
        h (torch.Tensor): entropy of x.
    """
    if norm:
        x = (x-x.min()) / (x.max()-x.min())
        x *= 255
    x = x.int() if torch.is_floating_point(x) else x

    output, counts = torch.unique(x, return_counts=True)
    norm_counts = counts / counts.sum()
    h = (-norm_counts * torch.log(norm_counts.float())).sum()

    import numpy as np
    if base == "nature":
        pass
    elif base == "binary":
        h /= np.log(2)  # To bits

    return h
