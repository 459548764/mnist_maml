#!/usr/bin/env python

import tensorflow as tf

from model import Model
from maml import Maml


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    with tf.Session() as sess:
        model = Model(sess)
        maml = Maml(model, x_train, y_train, x_test, y_test)
        maml.train(sess)


if __name__ == "__main__":
    main()
