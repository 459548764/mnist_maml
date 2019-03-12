"""Simple MAML implementation.

Based on algorithm 1 from:
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning
for fast adaptation of deep networks." Proceedings of the 34th International
Conference on Machine Learning-Volume 70. JMLR. org, 2017.

https://arxiv.org/pdf/1703.03400.pdf
"""
import numpy as np
import tensorflow as tf


class Maml:
    def __init__(self,
                 model,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 alpha=0.5,
                 beta=0.5,
                 task_size=10,
                 batch_size=200):
        self._model = model
        self._beta = beta
        self._alpha = alpha
        self._task_size = task_size
        self._batch_size = batch_size
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def train(self, sess):
        sess.run(tf.global_variables_initializer())
        done = False
        best_acc = 0.0
        while not done:
            theta = None
            theta_prime = []
            tasks = self._sample_tasks()
            for t_i in tasks:
                theta, grads = self._model.compute_params_and_grads(
                    t_i["x"], t_i["y"])
                theta_prime.append({
                    x: theta[x] - self._alpha * grads[x]
                    for x in theta if x in grads
                })

            sum_grads = None
            for t_i, theta_i in zip(tasks, theta_prime):
                self._model.assign_model_params(theta_i)
                _, grads = self._model.compute_params_and_grads(
                    t_i["x"], t_i["y"])
                if sum_grads is None:
                    sum_grads = grads
                else:
                    sum_grads = {
                        x: sum_grads[x] + grads[x]
                        for x in theta if x in grads
                    }

            theta = {
                x: theta[x] - self._beta * sum_grads[x]
                for x in theta if x in sum_grads
            }
            self._model.assign_model_params(theta)

            acc = self._model.compute_acc(self._x_test, self._y_test)
            print("Accuracy on test set {}".format(acc))

    def _sample_tasks(self):
        N = self._y_train.shape[0]
        tasks = []
        for _ in range(self._task_size):
            ids = np.random.choice(N, self._batch_size, False)
            tasks.append({"x": self._x_train[ids], "y": self._y_train[ids]})
        return tasks
