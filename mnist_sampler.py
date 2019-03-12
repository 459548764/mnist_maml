import numpy as np
from tensorflow.keras.datasets import mnist


class MnistSampler:
    def __init__(self, training_digits=None, batch_size=None):
        """Set the parameters to sample from MNIST.

        - training_digits: list of digits to use for training, e.g., if equal
          to [3, 4] only 3 and 4 digits are sampled for training.
        - batch_size: number of training instances to use when sampling from
          MNIST. If equal to None, the entire training set is used.
        """
        assert training_digits is None or (
            type(training_digits) == list
            and [0 <= train_dig <= 9 for train_dig in training_digits])

        if training_digits:
            # Remove any possible duplicates with set
            self._training_digits = list(set(training_digits))
        else:
            self._training_digits = list(range(10))

        (self._x_train, self._y_train), (self._x_test,
                                         self._y_test) = mnist.load_data()
        self._x_train = self._x_train / 255.0
        self._x_test = self._x_test / 255.0

        for i in range(10):
            if i not in self._training_digits:
                remove_train_indices = np.where(self._y_train == i)[0]
                self._x_train = np.delete(
                    self._x_train, remove_train_indices, axis=0)
                self._y_train = np.delete(
                    self._y_train, remove_train_indices, axis=0)
                remove_test_indices = np.where(self._y_test == i)[0]
                self._x_test = np.delete(
                    self._x_test, remove_test_indices, axis=0)
                self._y_test = np.delete(
                    self._y_test, remove_test_indices, axis=0)

        self._N = self._x_train.shape[0]
        assert batch_size is None or 0 <= batch_size <= self._N
        if batch_size:
            self._batch_size = batch_size
            self._sample_indices = np.array(range(0, self._N))
        else:
            self._batch_size = self._N

    def sample(self):
        if self._batch_size < self._N:
            end_of_epoch = False
            if self._sample_indices.size < self._batch_size:
                indices = self._sample_indices
                self._sample_indices = np.array(range(0, self._N))
                end_of_epoch = True
            else:
                indices = np.random.randint(0, len(self._sample_indices),
                                            self._batch_size)
                self._sample_indices = np.delete(self._sample_indices, indices)
            return self._x_train[indices], self._y_train[indices], end_of_epoch
        else:
            return self._x_train, self._y_train, True 

    def get_test_set(self):
        return self._x_test, self._y_test
