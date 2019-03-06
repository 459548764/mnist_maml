"""Simple MAML implementation.

Based on algorithm 1 from:
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning
for fast adaptation of deep networks." Proceedings of the 34th International
Conference on Machine Learning-Volume 70. JMLR. org, 2017.

https://arxiv.org/pdf/1703.03400.pdf
"""
import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, sess, model_name="model"):
        self._sess = sess
        self._x = None
        self._y = None
        self._out = None
        self._params = None
        self._grads = None
        self._acc = None
        self._name = model_name
        self._build_model()
        self._build_gradients()
        self._build_accuracy()

    def _build_model(self):
        # Model inputs
        self._x = tf.placeholder(shape=(None, 28, 28), dtype='float32')
        self._y = tf.placeholder(shape=(None, ), dtype='int64')
        # Model layers
        x_flat = tf.layers.Flatten()(self._x)
        with tf.variable_scope(self._name, values=[x_flat]):
            hidden_1 = tf.layers.dense(x_flat, 512, activation=tf.nn.relu)
            hidden_2 = tf.nn.dropout(hidden_1, rate=0.2)
            self._out = tf.layers.dense(hidden_2, 10, activation=tf.nn.softmax)

    def _build_gradients(self):
        loss = tf.losses.sparse_softmax_cross_entropy(self._y, self._out)
        adam_opt = tf.train.AdamOptimizer()
        grad_var = adam_opt.compute_gradients(loss)

        # Gradients where keys are the TF variable names
        self._grads = {}
        # Model parameters where keys are the TF variable names
        self._params = {}
        for grad, var in grad_var:
            self._grads.update({var.name: grad})
            self._params.update({var.name: var})

    def _build_accuracy(self):
        # Calculate accuracy
        y_pred = tf.math.argmax(self._out, axis=1)
        self._acc = tf.reduce_mean(
            tf.cast(tf.equal(y_pred, self._y), tf.float32))

    def compute_params_and_grads(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        return self._sess.run([self._params, self._grads], feed_dict=feed_dict)

    def compute_acc(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        return self._sess.run(self._acc, feed_dict=feed_dict)

    def assign_model_params(self, params):
        assign_ops = []
        for i in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name):
            if i.name in params:
                assign_ops.append(i.assign(params[i.name]))
        self._sess.run(assign_ops)


class Maml():
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
