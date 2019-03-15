#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers

from maml import Maml
from mnist_sampler import MnistSampler
from model import Model


def main():
    known_digits = list(range(7))
    tasks = [
        MnistSampler(known_digits + [7], batch_size=100),
        MnistSampler(known_digits + [8], batch_size=100),
        MnistSampler(known_digits + [9], batch_size=100)
    ]
    with tf.Session() as sess:
        model = Model([
            layers.Flatten(),
            layers.Dense(units=512, activation=tf.nn.relu),
            layers.Dropout(rate=0.2),
            layers.Dense(units=10, activation=tf.nn.softmax)
        ], {
            'shape': (None, 28, 28),
            'dtype': 'float32'
        }, {
            'shape': (None, ),
            'dtype': 'int64'
        }, sess)
        maml = Maml(model, tasks)
        maml.train(sess, 3)


if __name__ == "__main__":
    main()
