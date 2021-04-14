import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import sys
import os
from sklearn.metrics import mean_squared_error
from functools import wraps
import time


def time_it(function):
    @wraps(function)
    def runandtime(*args, **kwargs):
        tic = time.perf_counter()
        result = function(*args, **kwargs)
        toc = time.perf_counter()
        print(f'{function.__name__} took {toc-tic} seconds.')
        return result
    return runandtime

def generate_inputs(input=None, previous=None, predicted=None, axis=1):
    '''
    Generate starting/ending positions of q_total windows q of size q_size.
    Previous: where to start from, global. 1 means from the first patient visit.
    Predicted: the last (global) patient visit available to train (starting
    idx from 1).

     t is the current time (patient visit vector data). t_idx is
     the array index, so t_idx = t-1 .

    :param input: The input nparray
    :param axis: The axis to slice the input data.
    :param previous: the previous (t-1) input vector time (i.e. index)
    :param predicted: the current (t) input vector time (i.e. index)
    :return:
    '''
    # TODO: run tests to make sure that input CAN be sliced.
    # T_0 is the first time point.
    for t in range(previous, predicted):
        yield ( input[:, t-1, :], input [:, t, :], t, t+1)



def with_reproducible_rng(class_method):
    '''
    This is a function wrapper that calls rng.seed before every method call. Therefore user is expected to get the exact
    same results every time between different method calls of the same class instance.
    Multiple method calls in main file will produce the exact same results. This is of course if model parameters are
    the same between calls; changing e.g. cell no, will invoke different number of rand() called in each case.
    As seed, the network serial number is used.
    A function wrapper is used to dissociate different function calls: e.g. creating stimulus will be the same, even if
    user changed the number of rand() was called in a previous class function, resulting in the same stimulated cells.
    Multiple method calls in main file will produce the exact same results. This is of course if model parameters are
    the same between calls; changing e.g. cell no, will invoke different number of rand() called in each case.
    :param func:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self._epoch_completed ar runtime
        seed = args[0]._epoch_completed
        np.random.seed(seed)
        print(f'{class_method.__name__} reseeds the RNG with seed: {seed}.')
        return class_method(*args, **kwargs)
    return reset_rng

# transform each step of x, i.e. x_i into h_i
class Encoder(Model):
    def __init__(self, hidden_size):
        super().__init__(name='encode_share')
        self.hidden_size = hidden_size
        self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, input_x):
        sequence_time, c, h = input_x
        state = [c, h]
        output, state = self.LSTM_Cell_encode(sequence_time, state)
        return state[0], state[1]


# decode or generate the next sequence
class Decoder(Model):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode_share')
        self.hidden_size = hidden_size
        self.feature_dims = feature_dims
        self.LSTM_Cell_decode = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, input_x):
        sequence_time, encode_h, decode_c, decode_h = input_x
        decode_input = tf.concat((sequence_time, encode_h), axis=1)
        state = [decode_c, decode_h]
        output, state = self.LSTM_Cell_decode(decode_input, state)
        y_i = self.dense1(output)
        y_i = self.dense2(y_i)
        y_i = self.dense3(y_i)
        return y_i, state[0], state[1]


class HawkesProcess(Model):
    def __init__(self):
        super().__init__(name='point_process')

    def build(self, input_shape):
        shape_weight = tf.TensorShape((1, 1))
        self.trigger_parameter_alpha = self.add_weight(name='trigger_alpha',
                                                       shape=shape_weight,
                                                       initializer='uniform',
                                                       trainable=True)

        self.trigger_parameter_beta = self.add_weight(name='trigger_beta',
                                                      shape=shape_weight,
                                                      initializer='uniform',
                                                      trainable=True)

        self.base_intensity = self.add_weight(name='trigger_beta',
                                              shape=shape_weight,
                                              initializer='uniform',
                                              trainable=True)
        super(HawkesProcess, self).build(input_shape)

    def calculate_lambda_process(self, input_t, current_time_index, trigger_alpha, trigger_beta, base_intensity):
        batch = tf.shape(input_t)[0]
        current_t = tf.reshape(input_t[:, current_time_index], [batch, -1])
        current_t_tile = tf.tile(current_t, [1, current_time_index])

        time_before_t = input_t[:, :current_time_index]

        time_difference = time_before_t - current_t_tile

        trigger_kernel = tf.reduce_sum(tf.exp(trigger_beta * time_difference), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        condition_intensity_value = base_intensity + trigger_kernel * trigger_alpha
        return condition_intensity_value

    def calculate_likelihood(self, input_t, current_time_index, trigger_alpha, trigger_beta, base_intensity):
        batch = tf.shape(input_t)[0]
        ratio_alpha_beta = trigger_alpha / trigger_beta

        current_t = tf.reshape(input_t[:, current_time_index], [batch, 1])
        current_t_tile = tf.tile(current_t, [1, current_time_index])

        time_before_t = input_t[:, :current_time_index]

        # part_1: t_i -t(<i)
        time_difference = time_before_t - current_t_tile

        trigger_kernel = tf.reduce_sum(tf.exp(trigger_beta * time_difference), axis=1)
        trigger_kernel = tf.reshape(trigger_kernel, [batch, 1])

        conditional_intensity = base_intensity + trigger_alpha * trigger_kernel  # part 1 result

        # part_2: t_i - t_(i-1)
        last_time = input_t[:, current_time_index-1]
        time_difference_2 = (tf.reshape(last_time, [batch, 1]) - current_t) * base_intensity  # part 2 result

        # part_3: t_(i-1) - t(<i)
        last_time_tile = tf.tile(tf.reshape(last_time, [batch, 1]), [1, current_time_index])
        time_difference_3 = time_before_t - last_time_tile
        time_difference_3 = tf.reduce_sum(tf.exp(time_difference_3 * trigger_beta), axis=1)
        time_difference_3 = tf.reshape(time_difference_3, [batch, 1])

        probability_result = conditional_intensity * tf.exp(time_difference_2 + ratio_alpha_beta*(trigger_kernel - time_difference_3))

        return probability_result

    def call(self, input_x):
        input_t, current_time_index_shape = input_x
        current_time_index = tf.shape(current_time_index_shape)[0]
        batch = tf.shape(input_t)[0]
        trigger_alpha = tf.tile(self.trigger_parameter_alpha, [batch, 1])
        trigger_beta = tf.tile(self.trigger_parameter_beta, [batch, 1])
        base_intensity = tf.tile(self.base_intensity, [batch, 1])

        condition_intensity = self.calculate_lambda_process(input_t, current_time_index,
                                                            trigger_alpha, trigger_beta, base_intensity)
        likelihood = self.calculate_likelihood(input_t, current_time_index, trigger_alpha,
                                               trigger_beta, base_intensity)
        return condition_intensity, likelihood


def init_logger(name):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(name)

    print(path)
    print(os.path.dirname(__file__))
    print('------------------')


def split_dataset(dataset, k):
    train_set = [0] * k
    validate_set = [0] * k
    num_each_validate = int(dataset.shape[0] / k)

    for i in range(k):
        if i != k-1 and i != 0:
            validate_set[i] = dataset[i*num_each_validate:(i+1)*num_each_validate, :, :]
            train_set[i] = np.concatenate((dataset[:i*num_each_validate, :, :, ], dataset[(i+1)*num_each_validate:, :, :]), axis=0)

        if i == 0:
            validate_set[i] = dataset[:num_each_validate, :, :]
            train_set[i] = dataset[num_each_validate:, :, :]

        if i == k-1:
            validate_set[i] = dataset[(k-1)*num_each_validate:, :, :]
            train_set[i] = dataset[:(k-1)*num_each_validate, :, :]

    return train_set, validate_set


'''
if __name__== '__main__':
    train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    dataset = np.concatenate((train_set, test_set), axis=0)
    k = 5
    train_set, validate_set = split_dataset(dataset, k)
    for i in range(k):
        print(train_set[k].shape)
        print(validate_set[k].shape)
'''



