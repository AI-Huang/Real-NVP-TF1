#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-18-21 15:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import numpy as np
import tensorflow as tf


def parse_cmd_args():
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    hparams = parser.parse_args()  # So error if typo

    return hparams


def get_its(hparams):
    """Get number of training and validation iterations
    """
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hparams.n_train / hparams.n_batch_train))
    test_its = int(np.ceil(hparams.n_test / hparams.n_batch_train))
    train_epoch = train_its * hparams.n_batch_train

    # Do a full validation run
    full_test_its = hparams.n_test // hparams.local_batch_test

    return train_its, test_its, full_test_its


def get_data(hparams, sess):
    if hparams.image_size == -1:
        hparams.image_size = {'mnist': 32, 'cifar10': 32, 'imagenet-oord': 64,
                              'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hparams.problem]
    if hparams.n_test == -1:
        hparams.n_test = {'mnist': 10000, 'cifar10': 10000,
                          'imagenet-oord': 50000, 'imagenet': 50000}[hparams.problem]
    hparams.n_y = {'mnist': 10, 'cifar10': 10, 'imagenet-oord': 1000,
                   'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1}[hparams.problem]

    # Use anchor_size to rescale batch size based on image_size
    s = hparams.anchor_size
    hparams.local_batch_train = hparams.n_batch_train * \
        s * s // (hparams.image_size * hparams.image_size)
    hparams.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hparams.local_batch_train]  # round down to closest divisor of 50
    hparams.local_batch_init = hparams.n_batch_init * \
        s * s // (hparams.image_size * hparams.image_size)

    # datasets which's loading don't need sess
    if hparams.problem in ['mnist', 'cifar10']:
        hparams.direct_iterator = False
        import datasets.get_mnist_cifar as v

        train_iterator, test_iterator, data_init = \
            v.get_data(hparams.problem, hparams.dal, hparams.local_batch_train,
                       hparams.local_batch_test, hparams.local_batch_init, hparams.image_size)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def tensorflow_session():
    """Create tensorflow session with horovod
    """
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


def main(hparams):
    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hparams.seed)
    np.random.seed(hparams.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hparams, sess)
    hparams.train_its, hparams.test_its, hparams.full_test_its = get_its(
        hparams)

    # Create log dir
    logdir = os.path.abspath(hparams.logdir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Create model
    import models
    model = models.model(sess, hparams, train_iterator,
                         test_iterator, data_init)

    # Initialize visualization functions
    visualise = init_visualizations(hps, model, logdir)

    if not hparams.inference:
        # Perform training
        train(sess, model, hparams, logdir, visualise)
    else:
        infer(sess, model, hparams, test_iterator)


def train(sess, model, hps, logdir, visualise):
    _print(hps)
    _print('Starting training. Logging to', logdir)
    _print('epoch n_processed n_images ips dtrain dtest dsample dtot train_results test_results msg')

    # Train
    sess.graph.finalize()
    n_processed = 0
    n_images = 0
    train_time = 0.0
    test_loss_best = 999999

    if hvd.rank() == 0:
        train_logger = ResultLogger(logdir + "train.txt", **hps.__dict__)
        test_logger = ResultLogger(logdir + "test.txt", **hps.__dict__)

    tcurr = time.time()
    for epoch in tqdm(range(1, hps.epochs)):

        t = time.time()

        train_results = []
        for it in tqdm(range(hps.train_its)):

            # Set learning rate, linearly annealed from 0 in the first hps.epochs_warmup epochs.
            lr = hps.lr * min(1., n_processed /
                              (hps.n_train * hps.epochs_warmup))

            # Run a training step synchronously.
            _t = time.time()
            train_results += [model.train(lr)]  # 这里写train函数model.train
            if hps.verbose and hvd.rank() == 0:
                _print(n_processed, time.time()-_t, train_results[-1])
                sys.stdout.flush()

            # Images seen wrt anchor resolution
            n_processed += hvd.size() * hps.n_batch_train
            # Actual images seen at current resolution
            n_images += hvd.size() * hps.local_batch_train

        train_results = np.mean(np.asarray(train_results), axis=0)

        dtrain = time.time() - t
        ips = (hps.train_its * hvd.size() * hps.local_batch_train) / dtrain
        train_time += dtrain

        if hvd.rank() == 0:
            train_logger.log(epoch=epoch, n_processed=n_processed, n_images=n_images, train_time=int(
                train_time), **process_results(train_results))

        if epoch < 10 or (epoch < 50 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0:
            test_results = []
            msg = ''

            t = time.time()
            # model.polyak_swap()

            if epoch % hps.epochs_full_valid == 0:
                # Full validation run
                for it in range(hps.full_test_its):
                    test_results += [model.test()]
                test_results = np.mean(np.asarray(test_results), axis=0)

                if hvd.rank() == 0:
                    test_logger.log(epoch=epoch, n_processed=n_processed,
                                    n_images=n_images, **process_results(test_results))

                    # Save checkpoint
                    if test_results[0] < test_loss_best:
                        test_loss_best = test_results[0]
                        model.save(logdir+f"model_best_loss_epoch{epoch}.ckpt")
                        msg += ' *'

            dtest = time.time() - t

            # Sample
            t = time.time()
            if epoch == 1 or epoch == 10 or epoch % hps.epochs_full_sample == 0:
                visualise(epoch)
            dsample = time.time() - t

            if hvd.rank() == 0:
                dcurr = time.time() - tcurr
                tcurr = time.time()
                _print(epoch, n_processed, n_images, "{:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
                    ips, dtrain, dtest, dsample, dcurr), train_results, test_results, msg)

            # model.polyak_swap()

    if hvd.rank() == 0:
        _print("Finished!")


if __name__ == "__main__":
    hparams = parse_cmd_args()
    main(hparams)
