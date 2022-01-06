#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-15-21 15:13
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/docs/stable/_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_hparams

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


def add_hparams(
    writer, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
):
    """Add a set of hyperparameters to be compared in TensorBoard.

    Args:
        writer: a SummaryWriter object

        hparam_dict (dict): Each key-value pair in the dictionary is the
            name of the hyper parameter and it's corresponding value.
            The type of the value can be one of `bool`, `string`, `float`,
            `int`, or `None`.
        metric_dict (dict): Each key-value pair in the dictionary is the
            name of the metric and it's corresponding value. Note that the key used
            here should be unique in the tensorboard record. Otherwise the value
            you added by ``add_scalar`` will be displayed in hparam plugin. In most
            cases, this is unwanted.
        hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
            contains names of the hyperparameters and all discrete values they can hold
        run_name (str): Name of the run, to be included as part of the logdir.
            If unspecified, will use current timestamp.

    Examples::

        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter() as w:
            for i in range(5):
                w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

    Expected result:

    .. image:: _static/img/tensorboard/add_hparam.png
        :scale: 50 %

    """
    torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError(
            'hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict,
                            hparam_domain_discrete)

    # if not run_name:
    # Change it to below
    if run_name is None:  # "" is not None
        run_name = str(time.time())
    logdir = os.path.join(writer._get_file_writer().get_logdir(), run_name)
    with SummaryWriter(log_dir=logdir) as w_hp:
        w_hp.file_writer.add_summary(exp)
        w_hp.file_writer.add_summary(ssi)
        w_hp.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            w_hp.add_scalar(k, v)


def main():

    pass


if __name__ == "__main__":
    main()
