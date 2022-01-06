#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-05-22 09:17
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf
from models.real_nvp.real_nvp_loss import CompressionLoss
import tensorflow_probability as tfp


def main():
    dims = 32*32*3
    compression_loss = CompressionLoss(lambda_=500, dims=dims)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z = tf.random.uniform([64, dims], dtype=tf.float32)
        sldj = tf.random.uniform([64], dtype=tf.float32)
        loss = compression_loss.forward(z, sldj)

        loss = sess.run(loss)
        print(loss)


if __name__ == "__main__":
    main()
