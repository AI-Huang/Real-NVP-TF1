#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-16-21 22:53
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : http://example.org

import tensorflow as tf
from models.networks.resnet import ResidualBlock, ResNet


def test_residual_block():
    # Build graph first
    input_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_')
    block = ResidualBlock(filters=3, weight_norm=True,
                          name="residual_block_1", is_training=True)
    output = block.forward(input_)

    # Execute with a session
    with tf.Session() as sess:
        # Initialize variables
        # Same effect with below
        # tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())

        x_ = tf.random.uniform([64, 32, 32, 3], dtype=tf.float32)

        # {input_: x_}, 不要重名
        output = sess.run(output, feed_dict={input_: x_.eval()})

        print(type(output))
        print(output.shape)
        print("successfully run test_residual_block!")


def test_resnet():
    # Build graph first
    input_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_')
    resnet = ResNet(mid_channels=16, out_channels=16,
                    num_blocks=3, kernel_size=3, padding='valid', double_after_norm=True, name='resnet_1', is_training=True)
    output = resnet.forward(input_)

    # Execute with a session
    with tf.Session() as sess:
        # Initialize variables
        # Same effect with below
        # tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())

        x_ = tf.random.uniform([64, 32, 32, 3], dtype=tf.float32)

        # {input_: x_}, 不要重名
        output = sess.run(output, feed_dict={input_: x_.eval()})

        print(type(output))
        print(output.shape)
        print("successfully run test_resnet!")


def main():
    # test_residual_block()
    test_resnet()


if __name__ == "__main__":
    main()
