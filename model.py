import tensorflow as tf


class Model:
    def __init__(self, sess, model_name="model"):
        self._sess = sess
        self._x = None
        self._y = None
        self._out = None
        self._params = None
        self._grads = None
        self._acc = None
        self._optimize = None
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
        self._optimize = adam_opt.apply_gradients(grad_var)

    def _build_accuracy(self):
        # Calculate accuracy
        y_pred = tf.math.argmax(self._out, axis=1)
        self._acc = tf.reduce_mean(
            tf.cast(tf.equal(y_pred, self._y), tf.float32))

    def compute_params_and_grads(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        return self._sess.run([self._params, self._grads], feed_dict=feed_dict)

    def optimize(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        self._sess.run([self._optimize], feed_dict=feed_dict)

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
