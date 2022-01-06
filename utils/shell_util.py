#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-06-22 07:43
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : http://example.org

class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """

    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
