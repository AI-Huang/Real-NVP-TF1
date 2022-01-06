#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-06-22 07:43
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


import tensorflow as tf
from models.real_nvp.coupling_layer import CouplingLayerCompression, MaskType


def main():
    in_channels = 4
    mid_channels = 8
    num_blocks = 3
    mask_type = MaskType.CHANNEL_WISE
    reverse_mask = False
    share_weights = True

    # Build graph first
    input_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_')
    coupling_layer = CouplingLayerCompression(4 * in_channels,
                                              mid_channels,
                                              num_blocks,
                                              mask_type=mask_type,
                                              reverse_mask=reverse_mask,
                                              share_weights=share_weights)
    output = coupling_layer.forward(input_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_ = tf.random.uniform([64, 32, 32, 3], dtype=tf.float32)
        output = sess.run(output, feed_dict={input_: x_.eval()})

        output = sess.run(output)
        print(output)


if __name__ == "__main__":
    main()
