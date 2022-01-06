#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-17-21 22:13
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


"""
Ported from https://github.com/chrischute/real-nvp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.coupling_layer import CouplingLayer, CouplingLayerCompression, MaskType
from models.resnet import ResNet
from utils import squeeze_2x2, quantize


class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_steps (int): Number of flow steps in each in_couplings/out_couplings block in each flow level.
        share_weights (bool): Whether the s_net and t_net will share weights.
    """

    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks, num_steps, share_weights):
        super(_RealNVP, self).__init__()
        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([])
        mask_type = MaskType.CHECKERBOARD
        reverse_mask = False
        for _ in range(num_steps):
            self.in_couplings.append(
                CouplingLayer(in_channels,  mid_channels, num_blocks,
                              mask_type=mask_type,
                              reverse_mask=reverse_mask,
                              share_weights=share_weights)
            )
            # Reverse mask after each step
            reverse_mask = not reverse_mask

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks,
                              MaskType.CHECKERBOARD,
                              reverse_mask=True,
                              share_weights=share_weights)
            )
        else:
            self.out_couplings = nn.ModuleList([])
            mask_type = MaskType.CHANNEL_WISE
            reverse_mask = False
            for _ in range(num_steps):
                self.out_couplings.append(
                    CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks,
                                  mask_type=mask_type,
                                  reverse_mask=reverse_mask,
                                  share_weights=share_weights)
                )
                # Reverse mask after each step
                reverse_mask = not reverse_mask

            self.next_block = _RealNVP(
                scale_idx + 1, num_scales,
                2 * in_channels, 2 * mid_channels, num_blocks, num_steps, share_weights)

    def forward(self, x, sldj, reverse=False, **kwargs):
        """forward recursively
        """
        if not reverse:  # Not reverse ->
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

        else:  # reverse <-
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                # TODO, next_block should be "last_block"
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)

        return x, sldj


class _RealNVPCompressionNode(nn.Module):
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks, num_steps, share_weights):
        super(_RealNVPCompressionNode, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([])
        mask_type = MaskType.CHECKERBOARD
        reverse_mask = False
        for _ in range(num_steps):
            self.in_couplings.append(
                CouplingLayer(in_channels,  mid_channels, num_blocks,
                              mask_type=mask_type,
                              reverse_mask=reverse_mask,
                              share_weights=share_weights)
            )
            # Reverse mask after each step
            reverse_mask = not reverse_mask

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks,
                              MaskType.CHECKERBOARD,
                              reverse_mask=True,
                              share_weights=share_weights)
            )
        else:
            self.out_couplings = nn.ModuleList([])
            mask_type = MaskType.CHANNEL_WISE
            reverse_mask = False
            for _ in range(num_steps):
                self.out_couplings.append(
                    CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks,
                                  mask_type=mask_type,
                                  reverse_mask=reverse_mask,
                                  share_weights=share_weights)
                )
                # Reverse mask after each step
                reverse_mask = not reverse_mask

    def forward_obverse(self, x, sldj):
        """_RealNVPCompressionNode forward obversely, sequential implementation
        """
        for coupling in self.in_couplings:
            x, sldj = coupling(x, sldj, reverse=False)

        if not self.is_last_block:  # out_couplings
            # Squeeze -> 3x coupling (channel-wise)
            x = squeeze_2x2(x, reverse=False)
            for coupling in self.out_couplings:
                x, sldj = coupling(x, sldj, reverse=False)
            x = squeeze_2x2(x, reverse=True)

        return x, sldj

    def forward_reverse(self, x, sldj, x_split=None):
        """_RealNVPCompressionNode forward obversely, sequential implementation
        """
        if not self.is_last_block:
            x = torch.cat((x, x_split), dim=1)
            x = squeeze_2x2(x, reverse=True, alt_order=True)

            # Squeeze -> 3x coupling (channel-wise)
            x = squeeze_2x2(x, reverse=False)
            for coupling in reversed(self.out_couplings):
                x, sldj = coupling(x, sldj, reverse=True)
            x = squeeze_2x2(x, reverse=True)

        for coupling in reversed(self.in_couplings):
            x, sldj = coupling(x, sldj, reverse=True)

        return x, sldj

    def forward(self, x, sldj, x_split=None, reverse=False):
        """forward sequentially
        """
        if not reverse:  # Not reverse ->
            x, sldj = self.forward_obverse(x, sldj)
        else:  # reverse <-
            x, sldj = self.forward_reverse(x, sldj, x_split)

        return x, sldj


class _RealNVPCompression(nn.Module):
    """_RealNVPCompression, Sequential (not recursive) builder for a `RealNVPCompression` model.

    Each `_RealNVPCompressionBuilder` corresponds to a single scale in `RealNVPCompression`,
    and the constructor is recursively called to build a full `RealNVPCompression` model.

    Args:
        scale_idx (int): Index of current scale, starting from 0.
        num_scales (int): Number of scales in the RealNVPCompression model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_steps (int): Number of flow steps in each in_couplings/out_couplings block in each flow level.
        share_weights (bool): Whether the s_net and t_net will share weights.
    """

    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks, num_steps, share_weights):
        super(_RealNVPCompression, self).__init__()

        self.flow_blocks = nn.ModuleList([])
        for i in range(num_scales):
            block = _RealNVPCompressionNode(
                scale_idx+i, num_scales, in_channels*2**i,
                mid_channels*2**i, num_blocks, num_steps, share_weights)
            self.flow_blocks.append(block)

        self.phi_resnets = nn.ModuleList([])
        for i in reversed(range(num_scales)):
            resnet = ResNet(in_channels*2**i, mid_channels*2**i, in_channels*2**i,
                            num_blocks=num_blocks, kernel_size=3, padding=1,
                            double_after_norm=False)
            self.phi_resnets.append(resnet)

        self.zs_hat = None

    def forward_obverse(self, x, sldj):
        """forward-> forward obversely, sequential implementation
        """
        x_splits = []

        for idx, block in enumerate(self.flow_blocks):
            x, sldj = block.forward_obverse(x, sldj)
            if not block.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x_splits.append(x_split)  # Push into stack

        for x_split in reversed(x_splits):
            x = torch.cat((x, x_split), dim=1)
            x = squeeze_2x2(x, reverse=True, alt_order=True)

        return x, sldj

    def forward_reverse(self, x, sldj, if_quantize=False, factorized=False):
        """forward<- forward reversely, sequential implementation
        """
        zs_hat, x_splits = [], []
        # Make splits
        for _, block in enumerate(reversed(self.flow_blocks)):
            if not block.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                if if_quantize:
                    x_split = quantize(x_split)  # quantize before flow back
                x_splits.append(x_split)  # Store x_split of each node

        for idx, block in enumerate(reversed(self.flow_blocks)):
            if if_quantize:
                x = quantize(x)  # quantize before flow back
                zs_hat.append(x)

            if not block.is_last_block:

                if not factorized:
                    # Use x_split directly, when it's quantize branch
                    x_split = x_splits[-idx]
                else:
                    # Estimate x_split here, when it's sample branch
                    mu, var = x.mean(0, keepdim=True), x.var(
                        0, keepdim=True)  # sample
                    x_split = torch.cat((mu,) * x.size(0))
                    # TODO, var -> DLogistic
                    p_thresh = None

                x, sldj = block.forward_reverse(x, sldj, x_split)
                if if_quantize:
                    x = self.phi_resnets[idx](x)  # dequantize

            else:
                x, sldj = block.forward_reverse(x, sldj)
                if if_quantize:
                    x = self.phi_resnets[idx](x)  # dequantize

        zs_hat.append(x)  # Don't quantize at last output z

        self.zs_hat = zs_hat

        return x, sldj

    def forward(self, x, sldj, reverse=False, if_quantize=False, factorized=False):
        """forward sequentially
        """
        if not reverse:  # Not reverse ->
            x, sldj = self.forward_obverse(x, sldj)
        else:  # reverse <-
            x, sldj = self.forward_reverse(
                x, sldj, if_quantize=if_quantize, factorized=factorized)

        return x, sldj


class _RealNVPCompressionTrial1(nn.Module):
    """Recursive builder for a `RealNVP` model.
    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.
    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): E.g., 3, Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): C, Number of channels in the intermediate layers.
        num_blocks (int): B, Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_steps (int): K, Number of flow steps in each in_couplings/out_couplings block in each flow level.
        share_weights (bool): If True, the s_net and t_net will share the weights.
    """

    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks, num_steps, share_weights):
        super(_RealNVPCompressionTrial1, self).__init__()

        self.realnvp_layers = nn.ModuleList([])
        for _ in range(num_scales):
            self.realnvp_layers.append(self._make_layer(
                in_channels, mid_channels, num_blocks, num_steps, share_weights))

    def _make_layer(self, in_channels, mid_channels, num_blocks, num_steps, share_weights):
        """Make a Real NVP layer. Every Real NVP layer consists of ```num_steps``` of CouplingLayerCompression.
        """
        layers, reverse_mask = [], False
        mask_type = MaskType.CHANNEL_WISE
        for _ in range(num_steps):
            layers.append(
                CouplingLayerCompression(4 * in_channels,
                                         mid_channels,
                                         num_blocks,
                                         mask_type=mask_type,
                                         reverse_mask=reverse_mask,
                                         share_weights=share_weights)
            )
            # Reverse mask after each step
            reverse_mask = not reverse_mask
        return nn.Sequential(*layers)

    def forward(self, x, sldj, reverse=False, reverse_type="reconstruct"):
        """
        """
        delta = 1.0

        if not reverse:  # Not reverse ->
            for realnvp_layer in self.realnvp_layers:
                # Squeeze -> num_steps of coupling (channel-wise) -> unqueeze
                x = squeeze_2x2(x, reverse=False)
                for coupling in realnvp_layer:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

        else:  # reverse <-
            # reverse 传播有三种方式
            assert reverse_type in ["reconstruct", "quantize", "sample"]

            if reverse_type == "reconstruct":
                for realnvp_layer in reversed(self.realnvp_layers):
                    # Squeeze -> num_steps of coupling (channel-wise) -> unqueeze
                    x = squeeze_2x2(x, reverse=False)
                    for coupling in reversed(realnvp_layer):
                        x, sldj = coupling(x, sldj, reverse)
                    x = squeeze_2x2(x, reverse=True)

            elif reverse_type == "quantize":

                for realnvp_layer in reversed(self.realnvp_layers):
                    x = squeeze_2x2(x, reverse=False)
                    for coupling in reversed(realnvp_layer):
                        x, sldj = coupling(x, sldj, reverse)
                        u = (torch.rand(x.shape)-0.5) * delta
                        # quantize to z_hat
                        x = torch.round(x+u) - u
                    x = squeeze_2x2(x, reverse=True)

            elif reverse_type == "sample":
                for i, realnvp_layer in enumerate(reversed(self.realnvp_layers)):
                    x = squeeze_2x2(x, reverse=False)
                    for coupling in reversed(realnvp_layer):
                        x, sldj = coupling(x, sldj, reverse)
                        if i == 0:
                            # quantize last block to z_hat
                            u = (torch.rand(x.shape)-0.5) * delta
                            x = torch.round(x+u) - u
                    x = squeeze_2x2(x, reverse=True)

        return x, sldj


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper: "Density estimation using Real NVP" by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): default 2. Number of scales in the RealNVP model.
        in_channels (int): default 3. Number of channels in the input.
        mid_channels (int): default 64. Number of channels in the intermediate layers.
        num_blocks (int): B, default 8. Number of residual blocks in the s and t network of
        `Coupling` layers.
        num_steps (int): K, default 3. Number of flow steps in each in_couplings/out_couplings block in each flow level.
        real_nvp_model: class _RealNVP or class _RealNVPCompression.
        share_weights (bool): default True. If True, the s_net and t_net will share the weights.
    """

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, num_steps=3, real_nvp_model=_RealNVP, share_weights=True, ** kwargs):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor(
            [0.9], dtype=torch.float32))

        kwargs = {"scale_idx": 0,
                  "num_scales": num_scales,
                  "in_channels": in_channels, "mid_channels": mid_channels,
                  "num_blocks": num_blocks,
                  "num_steps": num_steps,
                  "share_weights": share_weights}

        self.flows = real_nvp_model(**kwargs)

    def forward(self, x, sldj=None, reverse=False, **kwargs):
        sldj = None
        if not reverse:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError(
                    f'Expected x in [0, 1], got x with min/max {x.min()}/{x.max()}')

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse, **kwargs)

        return x, sldj

    def reconstruct(self, x, if_quantize=False):
        # infer x by z
        z, sldj = self.forward(x, reverse=False)
        x_reconstruct, sldj = self.forward(
            z, reverse=True, if_quantize=if_quantize)
        return x_reconstruct

    def evaluate_mse(self, x, x_reconstruct=None, if_quantize=False):
        """ Evaluate reconstruction MSE
        Args:
            x_reconstruct: if given, F.mse_loss(x, x_reconstruct) is directly computed.
        """
        x_reconstruct = self.reconstruct(
            x, if_quantize) if x_reconstruct is None else x_reconstruct

        return F.mse_loss(x, x_reconstruct, reduction="mean")

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.data_constraint).log() -
                         self.data_constraint.log())
        sldj = ldj.view(ldj.size(0), -1).sum(-1)

        return y, sldj
