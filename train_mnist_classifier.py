#!/usr/bin/env python 

import tensorflow as tf

from model import Model
from mnist_sampler import MnistSampler


with tf.Session() as sess:
    model = Model(sess)
    sampler = MnistSampler(training_digits=list(range(8)), batch_size=200)
    sess.run(tf.global_variables_initializer())
    best_acc = 0.0
    num_epoch = 5
    end_of_epoch = False
    for itr in range(num_epoch):
        while not end_of_epoch:
            x_train, y_train, end_of_epoch = sampler.sample()
            model.optimize(x_train, y_train)
        x_test, y_test = sampler.get_test_set()
        it_acc = model.compute_acc(x_test, y_test)
        print("Epoch {}, accuracy {}".format(itr, it_acc))
        if it_acc > best_acc:
            best_acc = it_acc
    print("Best accuracy {}".format(it_acc))
