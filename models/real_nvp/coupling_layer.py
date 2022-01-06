#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-19-21 22:24
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

from enum import IntEnum
import numpy as np
import functools
import tensorflow as tf
from models.networks.resnet import ResNet, ResNetCompression
from utils import checkerboard_mask


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


# TODO change s_net(x_b) -> sess.run(s_net, TODO feed)

def coupling_forward(x, sldj=None, mask_type=MaskType.CHECKERBOARD, reverse=False, reverse_mask=False, share_weights=True, **kwargs):
    """Universal forward function for coupling layers.

    Args:
        x (tensor):
        sldj (tensor):
        mask_type:
        reverse:
        reverse_mask (bool): whether the mask should be reversed.
    kwargs:
        rescale (nn.Module):
        s_net (nn.Module):
        t_net (nn.Module):
        st_net (nn.Module):
    """
    # Optional layers
    rescale = kwargs["rescale"] if "rescale" in kwargs else None
    s_net = kwargs["s_net"] if "s_net" in kwargs else None
    t_net = kwargs["t_net"] if "t_net" in kwargs else None
    st_net = kwargs["st_net"] if "st_net" in kwargs else None
    if not share_weights:
        # s_net can be None, but t_net must exist
        assert not t_net == None
    else:
        assert not st_net == None

    if mask_type == MaskType.CHECKERBOARD:
        # Checkerboard mask
        b = checkerboard_mask(x.size(2), x.size(
            3), reverse_mask, device=x.device)
        x_b = x * b

        if not share_weights:
            s = s_net(x_b) if s_net else tf.ones(x_b.shape)
            t = t_net(x_b)
        else:
            st = st_net(x_b)
            s, t = st.chunk(2, dim=1)

        s = rescale(x=tf.tanh(s)) if rescale else s

        """Equation (9) from Density estimation using Real NVP (https://arxiv.org/abs/1605.08803).
        y=b\odot x + (1-b)\odot( x\odot \text{exp}(s(b\odot x)) + t(b\odot x) )
        """
        s, t = s * (1 - b), t * (1 - b)
        # Scale and translate
        if not reverse:
            exp_s = s.exp()
            if tf.math.is_nan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            # Original
            # x = (x + t) * exp_s
            x = x_b + (1 - b) * (x * exp_s + t)

            # Add log-determinant of the Jacobian
            sldj += s.reshape(s.size(0), -1).sum(-1)
        else:
            inv_exp_s = (-s).exp()
            if tf.math.is_nan(inv_exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')

            # Original
            # x = x * inv_exp_s - t
            x = x_b + (1 - b) * (x - t) * inv_exp_s

    else:
        # Channel-wise mask
        if not reverse_mask:
            x_change, x_id = x.chunk(2, dim=1)
        else:
            x_id, x_change = x.chunk(2, dim=1)

        if not share_weights:
            s, t = s_net(x_id) if s_net else tf.ones(x_id.shape), t_net(x_id)
        else:
            st = st_net(x_id)
            s, t = st.chunk(2, dim=1)

        s = rescale(x=tf.tanh(s)) if rescale else s

        # Scale and translate
        if not reverse:
            exp_s = s.exp()
            if tf.math.is_nan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            # TODO
            # x_change = (x_change + t) * exp_s
            x_change = x_change * exp_s + t

            # Add log-determinant of the Jacobian
            sldj += s.reshape(s.size(0), -1).sum(-1)
        else:
            inv_exp_s = s.mul(-1).exp()
            if tf.math.is_nan(inv_exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            # x_change = x_change * inv_exp_s - t
            x_change = (x_change - t) * inv_exp_s

        if not reverse_mask:
            x = tf.concat((x_change, x_id), axis=3)
        else:
            x = tf.concat((x_id, x_change), axis=3)

    return x, sldj


# class CouplingLayer(nn.Module):
#     """Coupling layer in RealNVP.

#     Args:
#         in_channels (int): Number of channels in the input.
#         mid_channels (int): Number of channels in the `s` and `t` network.
#         num_blocks (int): Number of residual blocks in the `s` and `t` network.
#         mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
#         reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
#         share_weights (bool): If True, the s_net and t_net will share the weights.
#     """

#     def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask, share_weights=False, **kwargs):
#         super(CouplingLayer, self).__init__()

#         # Save mask info
#         self.mask_type = mask_type
#         double_after_norm = (self.mask_type == MaskType.CHECKERBOARD)
#         self.reverse_mask = reverse_mask
#         self.share_weights = share_weights

#         # Build scale and translate network
#         if self.mask_type == MaskType.CHANNEL_WISE:
#             in_channels //= 2

#         if not share_weights:
#             self.s_net = ResNet(in_channels, mid_channels, in_channels,
#                                 num_blocks=num_blocks, kernel_size=3, padding=1,
#                                 double_after_norm=double_after_norm)
#             self.t_net = ResNet(in_channels, mid_channels, in_channels,
#                                 num_blocks=num_blocks, kernel_size=3, padding=1,
#                                 double_after_norm=double_after_norm)
#         else:
#             self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
#                                  num_blocks=num_blocks, kernel_size=3, padding=1,
#                                  double_after_norm=double_after_norm)

#         # Learnable scale for s
#         self.rescale = nn.utils.weight_norm(Rescale(in_channels))

#     def forward(self, x, sldj=None, reverse=False):
#         args = x, sldj, self.mask_type, reverse, self.reverse_mask, self.share_weights
#         kwargs = {"rescale": self.rescale}
#         if not self.share_weights:
#             kwargs.update({"s_net": self.s_net, "t_net": self.t_net})
#         else:
#             kwargs.update({"st_net": self.st_net})

#         x, sldj = coupling_forward(*args, **kwargs)

#         return x, sldj

def get_weight(shape, gain=np.sqrt(2), weight_norm=True, fan_in=None, name='weight'):
    """get_weight with weight_norm option.
    From https://github.com/alibaba-edu/simple-effective-text-matching/blob/master/src/modules/__init__.py
    """
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init

    w = tf.get_variable(name, shape=shape, initializer=tf.initializers.random_normal(0, std),
                        dtype=tf.float32)
    if weight_norm:
        g = tf.get_variable(f'{name}_g', shape=(1,) * (len(shape) - 1) + (shape[-1],),
                            initializer=tf.ones_initializer)
        w_norm = tf.sqrt(tf.reduce_sum(tf.square(w), axis=list(
            range(len(shape) - 1)), keep_dims=True))
        w = w / tf.maximum(w_norm, 1e-7) * g

    return w


def rescale(name, x, num_channels=None, weight_norm=True):
    """Per-channel rescaling.

    Args:
        x:
        num_channels (int): Number of channels in the input.
        weight_norm (bool): 
    """
    with tf.variable_scope(name):
        shape = x.shape
        w_shape = [1 for _ in range(
            len(shape)-2)]+[num_channels]

        w = get_weight(w_shape, weight_norm=weight_norm)

        output = tf.matmul(x, w)

        return output


class CouplingLayerCompression(object):
    """Coupling layer in RealNVP. Adapted to Image Compression

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
        share_weights (bool): If True, the s_net and t_net will share the weights.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask, share_weights=False, name=None, **kwargs):
        super(CouplingLayerCompression, self).__init__()
        assert name is not None
        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask
        self.share_weights = share_weights
        self.name = name

        # Build scale and translate network
        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels //= 2
        if not share_weights:
            self.s_net = None
            self.t_net = ResNetCompression(
                mid_channels, in_channels, num_blocks=num_blocks,
                name=self.name+".resnet_compression")
        else:
            self.st_net = ResNetCompression(
                mid_channels, 2 * in_channels, num_blocks=num_blocks,
                name=self.name+".resnet_compression")

        # Learnable scale for s
        # Use partial function
        self.rescale = functools.partial(rescale, name=self.name+".rescale",
                                         num_channels=mid_channels, weight_norm=True)

    def forward(self, x, sldj=None, reverse=True):
        args = x, sldj, self.mask_type, reverse, self.reverse_mask, self.share_weights
        kwargs = {"rescale": self.rescale}
        if not self.share_weights:
            kwargs.update({"s_net": self.s_net, "t_net": self.t_net})
        else:
            kwargs.update({"st_net": self.st_net})

        x, sldj = coupling_forward(*args, **kwargs)

        return x, sldj
