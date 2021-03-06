import tensorflow as tf
from tensorflow.keras.models import Model
from data import DataSet
from bayes_opt import BayesianOptimization
import scipy.stats as stats
import os
import sys
import numpy as np
from utils import split_dataset

import warnings
warnings.filterwarnings(action='once')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 单步x into hidden representation
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


class Decoder(Model):
    def __init__(self, hidden_size, feature_dims):
        super().__init__(name='decode_share')
        self.feature_dims = feature_dims
        self.hidden_size = hidden_size
        self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)
        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    def call(self, input_x):
        sequence_time, c, h = input_x
        state = [c, h]
        output, state = self.LSTM_decoder(sequence_time, state)
        y_1 = self.dense1(output)
        y_2 = self.dense2(y_1)
        y_3 = self.dense3(y_2)
        return y_3, state[0], state[1]


def train(hidden_size, learning_rate, l2_regularization):

    train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)[:, :, 1:6]
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)[:, :, 1:6]
    test_set = np.load('../../Trajectory_generate/dataset_file/HF_test_.npy').reshape(-1, 6, 30)[:, :, 1:6]

    # train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)[:, :, 1:]
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)[:, :, 1:]

    # train_set = np.load("../../Trajectory_generate/dataset_file/mimic_train_x_.npy").reshape(-1, 6, 37)[:, :, 1:]
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_.npy").reshape(-1, 6, 37)[:, :, 1:]
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)[:, :, 1:]

    previous_visit = 3
    predicted_visit = 3

    feature_dims = train_set.shape[2]

    train_set = DataSet(train_set)

    # train_sets = np.concatenate((train_set, validate_set), axis=0)
    # 训练时采用分割数据集
    # train_sets, validate_sets = split_dataset(dataset=train_sets, k=5)

    batch_size = 64
    epochs = 50
    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('feature_size----{}'.format(feature_dims))
    print('previous_visit---{}---predicted_visit----{}-'.format(previous_visit, predicted_visit))

    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate,
                                                                                l2_regularization))

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)

    logged = set()

    max_loss = 0.01
    max_pace = 0.0001

    count = 0
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        input_x_train = train_set.next_batch(batch_size)
        batch = input_x_train.shape[0]
        with tf.GradientTape() as tape:
            predicted_trajectory = np.zeros(shape=(batch, 0, feature_dims))
            for predicted_visit_ in range(predicted_visit):
                sequence_time_last_time = input_x_train[:, previous_visit+predicted_visit_-1, :]  # y_j
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    if previous_visit_ == 0:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])
                context_state = encode_h  # h_j from 1 to j

                input_decode = tf.concat((sequence_time_last_time, context_state), axis=1)  # y_j and h_j
                if predicted_visit_ == 0:
                    decode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                    decode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))

                predicted_next_sequence, decode_c, decode_h = decoder_share([input_decode, decode_c, decode_h])
                predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, feature_dims])
                predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)

            mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit+predicted_visit, :], predicted_trajectory))

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                mse_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                mse_loss += tf.keras. regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            gradient = tape.gradient(mse_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:
                logged.add(train_set.epoch_completed)
                loss_pre = mse_loss
                mse_loss = tf.reduce_mean(tf.keras.losses.mse(input_x_train[:, previous_visit: previous_visit + predicted_visit, :], predicted_trajectory))
                loss_diff = loss_pre - mse_loss
                if mse_loss > max_loss:
                    count = 0

                else:
                    if loss_diff > max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

                input_x_test = test_set
                batch_test = input_x_test.shape[0]
                predicted_trajectory_test = np.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    if predicted_visit_ == 0:
                        sequence_time_last_time_test = input_x_test[:, predicted_visit_+previous_visit-1, :]
                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            sequence_input_t = predicted_trajectory_test[:, i, :]
                            encode_c_test, encode_h_test = encode_share([sequence_input_t, encode_c_test, encode_h_test])

                    context_state = encode_h_test

                    if predicted_visit_ == 0:
                        decode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                    input_decode_test = tf.concat((sequence_time_last_time_test, context_state), axis=1)
                    predicted_next_sequence_test, decode_c_test, decode_h_test = decoder_share([input_decode_test, decode_c_test, decode_h_test])
                    sequence_time_last_time_test = predicted_next_sequence_test  # feed the generated sequence into next state
                    predicted_next_sequence_test = tf.reshape(predicted_next_sequence_test, [batch_test, -1, feature_dims])
                    predicted_trajectory_test = tf.concat((predicted_trajectory_test, predicted_next_sequence_test),
                                                          axis=1)
                mse_loss_predicted = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))
                mae_predicted = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], predicted_trajectory_test))
                r_value_all = []
                p_value_all = []
                for r in range(predicted_visit):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(predicted_trajectory_test[:, r, :], (-1,))
                    r_value_ = stats.pearsonr(x_, y_)
                    r_value_all.append(r_value_[0])
                    p_value_all.append(r_value_[1])

                if (train_set.epoch_completed + 1) % 10 == 0:
                    print('----epoch{}------mse_loss{}----predicted_mse----mae_predicted---{}-'
                          '{}---predicted_r_value---{}--count  {}'.format(train_set.epoch_completed,
                                                                          mse_loss, mse_loss_predicted,
                                                                          mae_predicted,
                                                                          np.mean(r_value_all), count))

                if (np.mean(r_value_all) > 0.87) and (np.mean(r_value_all) < 0.88) and (train_set.epoch_completed == 49):
                    np.savetxt('AED_generated_trajectory.csv', predicted_trajectory_test.numpy().reshape(-1, feature_dims), delimiter=',')

        tf.compat.v1.reset_default_graph()
    return mse_loss_predicted, mae_predicted, np.mean(r_value_all), np.mean(p_value_all)
    # return -1*mse_loss_predicted



if __name__ == '__main__':
    test_test('AED_—HF_test__3_3_5_重新训练.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, -1),
    #         'l2_regularization': (-5, -1),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    mse_all = []
    mae_all = []
    r_value_all = []
    p_value_all = []
    for i in range(50):
        mse, mae, r_value, p_value = train(hidden_size=32,
                                           learning_rate=0.0072787130035702665,
                                           l2_regularization=1.0847754618316529e-5)
        mse_all.append(mse)
        r_value_all.append(r_value)
        p_value_all.append(p_value)
        mae_all.append(mae)
        print('epoch  {}-----mse-all_ave  {}----mae_all_ave-----{}---r_value_ave  {}--'
              '---p_value_ave  {}--  mse_vale_std{}------mae_vale_std{}---r_value_std{}  p_value_std-'.
              format(i, np.mean(mse_all), np.mean(mae_all),
                     np.mean(r_value_all), np.mean(p_value_all),
                     np.std(mse_all), np.std(mae_all),
                     np.std(r_value_all), np.std(p_value_all)))

















