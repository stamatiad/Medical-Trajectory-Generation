import tensorflow as tf
import numpy as np
from data import DataSet
import os,sys
import warnings
from tensorflow.keras.models import Model
from test import Post, Prior, HawkesProcess, Encoder, Decoder
from scipy import stats
from bayes_opt import BayesianOptimization
import utils
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings(action='once')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def unstack_to_mat(df, nfeatures):
    # unstack getting the dims from args, because df is giving me hard time.
    m = int(df.shape[0]/nfeatures)
    n = int(nfeatures)
    #m, n = len(df.index.levels[-1]), len(df.index.levels[1])
    arr = df.values.reshape(m, n, -1).swapaxes(1, 2)
    return arr

class Discriminator(Model):
    def __init__(self, hidden_size, previous_visit, predicted_visit):
        super().__init__(name='discriminator')
        self.hidden_size = hidden_size
        self.previous_visit = previous_visit
        self.predicted_visit = predicted_visit

        self.dense1 = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        self.LSTM_Cell = tf.keras.layers.LSTMCell(hidden_size)

    def call(self, real_input, fake_input):
        batch = tf.shape(real_input)[0]

        input_same = real_input[:, :self.previous_visit, :]
        input_same_real = input_same
        input_same_fake = input_same

        trajectory_real = []
        trajectory_fake = []

        trajectory_real_predict = tf.zeros(shape=[batch, 0, 1])
        trajectory_fake_predict = tf.zeros(shape=[batch, 0, 1])
        for index in range(self.predicted_visit):
            next_real = real_input[:, index + self.previous_visit, :]
            next_fake = fake_input[:, index, :]
            next_real = tf.reshape(next_real, [batch, 1, -1])
            next_fake = tf.reshape(next_fake, [batch, 1, -1])
            trajectory_step_real = tf.concat((input_same_real, next_real), axis=1)
            trajectory_step_fake = tf.concat((input_same_fake, next_fake), axis=1)

            trajectory_real.append(trajectory_step_real)
            trajectory_fake.append(trajectory_step_fake)

            input_same_real = trajectory_step_real
            input_same_fake = trajectory_step_fake

        for time_index in range(self.predicted_visit):
            output_real = None
            output_fake = None
            trajectory_real_ = trajectory_real[time_index]
            trajectory_fake_ = trajectory_fake[time_index]

            state = self.LSTM_Cell.get_initial_state(batch_size=batch, dtype=tf.float32)
            state_real = state
            state_fake = state
            for t in range(tf.shape(trajectory_real_)[1]):
                input_real = trajectory_real_[:, t, :]
                input_fake = trajectory_fake_[:, t, :]
                output_real, state_real = self.LSTM_Cell(input_real, state_real)
                output_fake, state_fake = self.LSTM_Cell(input_fake, state_fake)

            output_fake = self.dense1(output_fake)
            output_real = self.dense1(output_real)

            trajectory_step_real_pre = self.dense2(output_real)
            trajectory_step_fake_pre = self.dense2(output_fake)

            trajectory_step_real_pre = self.dense3(trajectory_step_real_pre)
            trajectory_step_fake_pre = self.dense3(trajectory_step_fake_pre)

            trajectory_step_real_pre = self.dense4(trajectory_step_real_pre)
            trajectory_step_fake_pre = self.dense4(trajectory_step_fake_pre)

            trajectory_step_real_pre = tf.reshape(trajectory_step_real_pre, [batch, -1, 1])
            trajectory_step_fake_pre = tf.reshape(trajectory_step_fake_pre, [batch, -1, 1])

            trajectory_real_predict = tf.concat((trajectory_real_predict, trajectory_step_real_pre), axis=1)
            trajectory_fake_predict = tf.concat((trajectory_fake_predict, trajectory_step_fake_pre), axis=1)

        return trajectory_real_predict, trajectory_fake_predict


def train(hidden_size, z_dims, l2_regularization, learning_rate, n_disc, generated_mse_imbalance, generated_loss_imbalance, kl_imbalance, reconstruction_mse_imbalance, likelihood_imbalance):
    # train_set = np.load("../../Trajectory_generate/dataset_file/train_x_.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/test_x.npy").reshape(-1, 6, 60)
    # test_set = np.load("../../Trajectory_generate/dataset_file/validate_x_.npy").reshape(-1, 6, 60)

    # train_set = np.load('../../Trajectory_generate/dataset_file/HF_train_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_validate_.npy').reshape(-1, 6, 30)
    # test_set = np.load('../../Trajectory_generate/dataset_file/HF_test_.npy').reshape(-1, 6, 30)

    #train_set = np.load("../../Trajectory_generate/dataset_file
    # /mimic_train_x_.npy").reshape(-1, 6, 37)
    #test_set = np.load("../../Trajectory_generate/dataset_file/mimic_test_x_
    # .npy").reshape(-1, 6, 37)
    # test_set = np.load("../../Trajectory_generate/dataset_file/mimic_validate_.npy").reshape(-1, 6, 37)

    # Load (some) raw data:
    raw_data = pd.read_spss('BOUNCE_Dataset_m0_m3_m6_stefanos.sav')

    #  Some debugging: get only the columns you need:
    all_cols = raw_data.columns.values

    # Get M0/M3 as train and M6 as test:
    re_patern = re.compile('^M[0,3,6].*')
    vals, counts = np.unique(
        [i[3:] for i in all_cols if re_patern.match(i)],
        return_counts=True
    )
    valid_cols_idx = np.where(counts == 3, True, False)
    valid_cols = [i for i in all_cols if i[3:] in vals[valid_cols_idx] and
                  re_patern.match(i)]
    # TODO: any other way to split the features, making sure that they are
    #  sorted? I feel uneasy to do TWO sorts.
    features = vals[valid_cols_idx]
    features.sort()
    total_features = len(features)
    patient_no = raw_data.shape[0]
    # make sure that the features across time points (months) are consistent!
    valid_cols.sort()
    data_tmp = raw_data[valid_cols]

    # Transform the data in order to have the time frame also. Dims are:
    # patients, features, timepoints
    data_mat_tmp = np.transpose(
        data_tmp.values.reshape(-1, 3, total_features),
        (0, 2, 1)
    )
    # Now to make them Dataframe, MultiIndex compatible you need:
    data_mat = data_mat_tmp.reshape( -1, 3)
    patients_d = { i:f"Patient_{i}" for i in range(patient_no)}
    #patients_arr = np.arange(0, len(raw_data))
    timepoints_d = { i:f"M_{i}" for i in range(0, 9, 3)}
    #timepoints_arr = np.arange(0, 3)
    features_arr = features
    feature_dims = len(features_arr)

    midx = pd.MultiIndex.from_product([patients_d.values(), features_arr])
    # Finally create the multidimentional table (index in pandas) with all
    # the data. This makes it easy to handle. The testing/training functions
    # will also handle it well.
    data_all = pd.DataFrame(data_mat, index=midx, columns=timepoints_d.values())

    # Fill the NaN values with a value that does not appear naturally in the
    # input to signify a missing value (Can I do this?).
    data_all.fillna(-1, inplace=True)

    # Divide to train/test set: (basic up to this phase):
    # Do this with sklearn:
    split = [0.8, 0.2]
    split_seed = 123

    # Try to visualize/white the data to get a grip:

    for feature in features_arr:
        fig, ax = plt.subplots()
        blah = data_all.xs(feature, level=1, drop_level=False)
        feature_1 = blah['M_0'].values
        feature_2 = blah['M_3'].values
        feature_3 = blah['M_6'].values
        all_feats = np.concatenate((feature_1, feature_2, feature_3))
        bins = np.linspace(all_feats.min(), all_feats.max(), 20)
        histo, _ = np.histogram(feature_1, bins)
        ax.plot(bins[1:], histo, color='C0', label='M0')
        histo, _ = np.histogram(feature_2, bins)
        ax.plot(bins[1:], histo, color='C1', label='M3')
        histo, _ = np.histogram(feature_3, bins)
        ax.plot(bins[1:], histo, color='C2', label='M6')
        plt.legend()
        #plt.show()
        plt.savefig(f'Feature_{feature}.png')
        plt.close()

    # This requires df to be sorted, Why it is not?
    #blah = data_all.loc[(slice(None), slice('Apetite_QLQ30')), :]

    train_df, test_df = train_test_split(
        data_all, train_size=split[0],
        test_size=split[1],
        shuffle=False,
        #random_state=split_seed
    )

    #train_df.reindex()

    # This is the starting visit that the algorithm sees. I set it to 0,
    # since it is the first ever patient checkpoint/admission (month 0 in the
    # dataset).
    previous_visit = 0
    # This is the future admissions that the algorithm will predict. I set it
    # to 2 since this is the available ones that I have and I want to
    # maximally utilize my data (corresponds to M3, M6 admissions).
    predicted_visit = 2

    #m, n = len(df.index.levels[-1]), len(df.index.levels[1])

    # Cast data to tensors:
    train_mat = tf.constant(
        unstack_to_mat(train_df, feature_dims),
        dtype=tf.float16
    )

    train_set = DataSet(train_mat)
    train_set.epoch_completed = 0
    batch_size = 64
    epochs = 50

    # hidden_size = 2**(int(hidden_size))
    # z_dims = 2**(int(z_dims))
    # l2_regularization = 10 ** l2_regularization
    # learning_rate = 10 ** learning_rate
    # n_disc = int(n_disc)
    # generated_mse_imbalance = 10 ** generated_mse_imbalance
    # generated_loss_imbalance = 10 ** generated_loss_imbalance
    # kl_imbalance = 10 ** kl_imbalance
    # reconstruction_mse_imbalance = 10 ** reconstruction_mse_imbalance
    # likelihood_imbalance = 10 ** likelihood_imbalance

    print(f'feature_dims---{feature_dims}')

    print(f'previous_visit---{previous_visit}---predicted_visit----'
          f'{predicted_visit}-')

    print(f'hidden_size---{hidden_size}---z_dims---'
          f'{z_dims}---l2_regularization---{z_dims}'
          f'---learning_rate---{learning_rate}--n_disc---{n_disc}-'
          f'generated_mse_imbalance---'
          f'{generated_mse_imbalance}---generated_loss_imbalance---'
          f'{generated_loss_imbalance}---'
          f'kl_imbalance---{kl_imbalance}---reconstruction_mse_imbalance---'
          f'{reconstruction_mse_imbalance}---'
          f'likelihood_imbalance---{likelihood_imbalance}')

    encode_share = Encoder(hidden_size=hidden_size)
    decoder_share = Decoder(hidden_size=hidden_size, feature_dims=feature_dims)
    discriminator = Discriminator(predicted_visit=predicted_visit, hidden_size=hidden_size, previous_visit=previous_visit)

    post_net = Post(z_dims=z_dims)
    prior_net = Prior(z_dims=z_dims)

    hawkes_process = HawkesProcess()


    loss = 0
    count = 0
    optimizer_generation = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer_discriminator = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    logged = set()
    max_loss = 0.001
    max_pace = 0.0001

    while train_set.epoch_completed < epochs:
        input_train = train_set.next_batch(batch_size=batch_size)
        # It the second dim is the time, what is this?? Should be swapped; Also
        # from the usage of var generated_trajectory I also deduce that the
        # second dim is the time axis!!!
        # After reconsideration the second dim is the features (makes sense
        # to me). Don't sweat it; do it again.

        input_x_train = input_train
        #input_x_train = tf.cast(input_train[:, 1:, :], tf.float32)
        # This input_t_train should be the times of admissions. Should be:
        # batch_size x times of admissions.
        #input_t_train = tf.cast(input_train[:, 0, :], tf.float32)
        input_t_train = tf.constant(
            np.repeat(
                np.arange(
                    previous_visit,
                    (previous_visit + predicted_visit) * 3 +1 ,
                    3
                ),
                input_x_train.shape[0]
            ).reshape(
            input_x_train.shape[0], 3, order='F'),
            dtype=tf.float16
        )
        batch = input_train.shape[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            generated_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            probability_likelihood = tf.zeros(shape=[batch, 0, 1])
            reconstructed_trajectory = tf.zeros(shape=[batch, 0, feature_dims])
            z_mean_post_all = tf.zeros(shape=[batch, 0, z_dims])
            z_log_var_post_all = tf.zeros(shape=[batch, 0, z_dims])
            z_mean_prior_all = tf.zeros(shape=[batch, 0, z_dims])
            z_log_var_prior_all = tf.zeros(shape=[batch, 0, z_dims])
            # Multiple things I do not understand:
            # 1. If the user put unexpected values this loop will be skipped
            # altogether, so the generator should fail?
            # 2. I can not understand the previous_visit variable range. If
            # must it start from 0 then the loop will fail (first range
            # produced value will always result in index -1). So it always
            # gets the previous point from the point given by the user??

            # previous_visit should be >=1 or else we skip encoding altogether:
            # I don't know the algorithm on the original author's data,
            # but in my case it does not make any sense to have 0 diff to h_i
            # and h_i_1 because of NaN and 0 on LSTM h. So I adapt the code
            # accordingly.

            # TODO: This should always get initialized!
            decode_c_generate = tf.Variable(
                tf.zeros(shape=[batch, hidden_size]))
            decode_h_generate = tf.Variable(
                tf.zeros(shape=[batch, hidden_size]))

            decode_c_reconstruction = tf.Variable(
                tf.zeros(shape=[batch, hidden_size]))
            decode_h_reconstruction = tf.Variable(
                tf.zeros(shape=[batch, hidden_size]))
            # TODO: rewrite this in more pythonic way.
            # TODO: My understanding is that this should be only one
            #  iteration (i.e. no for loop) in my case, since I only need to
            #  predict the next patient state. ( Not necessary only the next
            #  one, yet I can only train with only the next one, due to
            #  limited timepoints in the dataset).
            for predicted_visit_ in range(1, predicted_visit):
                sequence_last_time = \
                    input_x_train[:, previous_visit + predicted_visit_ , :]
                sequence_current_time = \
                    input_x_train[:, previous_visit+predicted_visit_ +1, :]
                # Also struggling to understand:
                # Can I write this as an single feed to LSTM?

                # previous_visit_ will always be 0 at the beginning of the loop:
                encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                # TODO: rewrite this in more pythonic way.
                for previous_visit_ in range(previous_visit+predicted_visit_):
                    sequence_time = input_x_train[:, previous_visit_, :]
                    # why we need to feed each one of the time slices and hot
                    # the whole sequence?? Does not the code handles the
                    # whole sequence?
                    encode_c, encode_h = encode_share([sequence_time, encode_c, encode_h])

                encode_h_i = encode_h  # h_i
                encode_c, encode_h_i_1 = \
                    encode_share([sequence_current_time, encode_c, encode_h_i]) #
                # h_(i+1)


                z_post, z_mean_post, z_log_var_post = \
                    post_net([encode_h_i, encode_h_i_1])
                z_prior, z_mean_prior, z_log_var_prior = \
                    prior_net(encode_h_i)

                # Prospa8w na katalabw ti ginetai me to previous visit. Den
                # einai apla to starting point? Edw blepw oti to
                # current_time_index_shape einai megalytero apo to diff
                # previous-predicted pou 8a perimena...
                # As ftasw mexri edw k blepw meta..
                # TODO: make the HP returning a uniform conditional intensity
                #  function lambda(t). This is because we do not need the
                #  extra modeling power of the HP, since our admission
                #  intervals are uniformly spread checkpoints.

                # TODO: pythonize this mess:
                current_time_index_shape = \
                    tf.ones(shape=[previous_visit+predicted_visit_])
                # TODO: pythonize this mess:
                # TODO: does HP handle the larger input_t_train array? Yes
                #  BUT I don't understand the way.
                condition_value, likelihood = \
                    hawkes_process([input_t_train, current_time_index_shape])
                #TODO: Here likelihood is for a single time point. Is this
                # valid? Based on the author values I should not get more
                # than a single value?? What is going on?
                # Why here author uses tuple instead of list as argument?
                probability_likelihood = \
                    tf.concat(
                        (probability_likelihood, tf.reshape(likelihood, [batch, -1, 1])),
                        axis=1
                    )
                probability_likelihood = \
                    tf.keras.activations.softmax(probability_likelihood)
                # TODO: for what time points are the generation and
                #  reconstruction respectively?
                # TODO: pythonize this mess:
                generated_next_visit, decode_c_generate, decode_h_generate = \
                decoder_share(
                    [z_prior, encode_h_i, sequence_last_time, decode_c_generate, decode_h_generate*condition_value]
                )
                # reconstruction
                # TODO: pythonize this mess:
                reconstructed_next_visit, decode_c_reconstruction,
                decode_h_reconstruction = \
                    decoder_share(
                        [z_post, encode_h_i, sequence_last_time,
                                   decode_c_reconstruction,
                                   decode_h_reconstruction*condition_value]
                    )

                reconstructed_trajectory = \
                    tf.concat((
                    reconstructed_trajectory, tf.reshape(reconstructed_next_visit, [batch, -1, feature_dims])), axis=1)

                generated_trajectory = tf.concat((generated_trajectory, tf.reshape(generated_next_visit, [batch, -1, feature_dims])), axis=1)

                z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, z_dims])), axis=1)
                z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, z_dims])), axis=1)

                z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, z_dims])), axis=1)
                z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, z_dims])), axis=1)

            # Disable discriminator as it does not provide much of
            # performance (as by the authors). I can only use the MSE (Lr
            # error in the paper).

            generated_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    generated_trajectory))
            reconstructed_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train[:, previous_visit:previous_visit + predicted_visit, :],
                                    reconstructed_trajectory))

            std_post = tf.math.sqrt(tf.exp(z_log_var_post_all))
            std_prior = tf.math.sqrt(tf.exp(z_log_var_prior_all))

            kl_loss_element = 0.5 * (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(
                tf.maximum(std_post, 1e-9))
                                     + (tf.square(std_post) + (tf.square(z_mean_post_all - z_mean_prior_all)) /
                                        (tf.maximum(tf.square(std_prior), 1e-9))) - 1)
            kl_loss = tf.reduce_mean(kl_loss_element)

            likelihood_loss = tf.reduce_mean(probability_likelihood)

            loss += generated_mse_loss * generated_mse_imbalance +\
                    reconstructed_mse_loss * reconstruction_mse_imbalance + \
                    kl_loss * kl_imbalance + \
                    likelihood_loss * likelihood_imbalance

            variables = [var for var in encode_share.trainable_variables]
            for weight in encode_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            for weight in decoder_share.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in post_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in prior_net.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

            for weight in hawkes_process.trainable_variables:
                loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                variables.append(weight)

        '''
        for disc in range(n_disc):
            gradient_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
            optimizer_discriminator.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
        '''

        gradient_gen = gen_tape.gradient(loss, variables)
        optimizer_generation.apply_gradients(zip(gradient_gen, variables))

        # Why does the author loads the weights? Should he be saving them
        # instead? Also the modulo operator is ALWAYS 0 so why is there?
        # Also authors keep only the first batch of each epoch logged???

        #if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed
        # not in logged:
        if True:
            #encode_share.load_weights('encode_share_3_3_mimic.h5')
            #decoder_share.load_weights('decode_share_3_3_mimic.h5')
            #post_net.load_weights('post_net_3_3_mimic.h5')
            #prior_net.load_weights('prior_net_3_3_mimic.h5')
            #hawkes_process.load_weights('hawkes_process_3_3_mimic.h5')

            #logged.add(train_set.epoch_completed)
            loss_pre = generated_mse_loss

            mse_generated = tf.reduce_mean(
                tf.keras.losses.mse(
                    input_x_train[:, previous_visit:previous_visit+predicted_visit, :],
                    generated_trajectory
                )
            )
            mae_generated = tf.reduce_mean(
                tf.keras.losses.mae(
                    input_x_train[:, previous_visit:previous_visit+predicted_visit, :],
                    generated_trajectory
                )
            )

            print(f"\t\tBATCHID = {train_set.batch_completed} MSE ="
                  f" {mse_generated}")
            print(f"\t\tBATCHID = {train_set.batch_completed} MAE ="
                  f" {mae_generated}")

            loss_diff = loss_pre - mse_generated

            if mse_generated > max_loss:
                count = 0
            else:
                if loss_diff > max_pace:
                    count = 0
                else:
                    count += 1
            if count > 9:
                break

            # I guess this is the equivalent of an Cross Validation.
            #TODO: start with no CV: just report the training error and check
            # that it reduces. Them move to CV testing as below.
            if False:
                input_x_test = tf.cast(test_set[:, :, 1:], tf.float32)
                input_t_test = tf.cast(test_set[:, :, 0], tf.float32)

                batch_test = test_set.shape[0]
                generated_trajectory_test = tf.zeros(shape=[batch_test, 0, feature_dims])
                for predicted_visit_ in range(predicted_visit):
                    for previous_visit_ in range(previous_visit):
                        sequence_time_test = input_x_test[:, previous_visit_, :]
                        if previous_visit_ == 0:
                            encode_c_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                            encode_h_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))

                        encode_c_test, encode_h_test = encode_share([sequence_time_test, encode_c_test, encode_h_test])

                    if predicted_visit_ != 0:
                        for i in range(predicted_visit_):
                            encode_c_test, encode_h_test = encode_share([generated_trajectory_test[:, i, :], encode_c_test, encode_h_test])

                    context_state_test = encode_h_test

                    if predicted_visit_ == 0:
                        decode_c_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        decode_h_generate_test = tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
                        sequence_last_time_test = input_x_test[:, previous_visit+predicted_visit_-1, :]

                    z_prior_test, z_mean_prior_test, z_log_var_prior_test = prior_net(context_state_test)
                    current_time_index_shape_test = tf.ones([previous_visit+predicted_visit_])
                    intensity_value_test, likelihood_test = hawkes_process([input_t_test, current_time_index_shape_test])

                    generated_next_visit_test, decode_c_generate_test, decode_h_generate_test = decoder_share([z_prior_test, context_state_test, sequence_last_time_test, decode_c_generate_test, decode_h_generate_test*intensity_value_test])
                    generated_trajectory_test = tf.concat((generated_trajectory_test, tf.reshape(generated_next_visit_test, [batch_test, -1, feature_dims])), axis=1)
                    sequence_last_time_test = generated_next_visit_test

                mse_generated_test = tf.reduce_mean(tf.keras.losses.mse(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))
                mae_generated_test = tf.reduce_mean(tf.keras.losses.mae(input_x_test[:, previous_visit:previous_visit+predicted_visit, :], generated_trajectory_test))

                r_value_all = []
                p_value_all = []

                for r in range(predicted_visit):
                    x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
                    y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
                    if (y_.numpy() == np.zeros_like(y_)).all():
                        r_value_ = [0.0, 0.0]
                    else:
                        r_value_ = stats.pearsonr(x_, y_)
                    r_value_all.append(r_value_[0])
                    p_value_all.append(r_value_[1])

                print('epoch ---{}---train_mse_generated---{}---likelihood_loss{}---'
                      'train_mse_reconstruct---{}---train_kl---{}---'
                      'test_mse---{}---test_mae---{}---'
                      'r_value_test---{}---count---{}'.format(train_set.epoch_completed, generated_mse_loss, likelihood_loss,
                                                              reconstructed_mse_loss, kl_loss,
                                                              mse_generated_test, mae_generated_test,
                                                              np.mean(r_value_all), count))

                # if np.mean(r_value_all) > 0.9355:
                #     np.savetxt('generated_trajectory_test.csv', generated_trajectory_test.numpy().reshape(-1, feature_dims), delimiter=',')
                #     print('保存成功！')

                # if mse_generated_test < 0.0107:
                #     encode_share.save_weights('encode_share_3_3_mimic.h5')
                #     decoder_share.save_weights('decode_share_3_3_mimic.h5')
                #     discriminator.save_weights('discriminator_3_3_mimic.h5')
                #     post_net.save_weights('post_net_3_3_mimic.h5')
                #     prior_net.save_weights('prior_net_3_3_mimic.h5')
                #     hawkes_process.save_weights('hawkes_process_3_3_mimic.h5')
                #     print('保存成功！')


    sys.exit(0)
    tf.compat.v1.reset_default_graph()
    #TODO: Initially do not use CV in training.
    #return mse_generated_test, mae_generated_test, np.mean(r_value_all)
    return mse_generated, mae_generated, 0.0
    # return -1 * mse_generated_test


if __name__ == '__main__':
    # Start logger:
    utils.init_logger('blah.log')
    # BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'z_dims': (5, 8),
    #         'n_disc': (1, 10),
    #         'learning_rate': (-5, 1),
    #         'l2_regularization': (-5, 1),
    #         'kl_imbalance':  (-6, 1),
    #         'reconstruction_mse_imbalance': (-6, 1),
    #         'generated_mse_imbalance': (-6, 1),
    #         'likelihood_imbalance': (-6, 1),
    #         'generated_loss_imbalance': (-6, 1),
    #
    #     }
    # )
    # BO.maximize()
    # print(BO.max)

    mse_all = []
    r_value_all = []
    mae_all = []
    for i in range(50):
        mse, mae, r_value = train(hidden_size=32,
                                  z_dims=64,
                                  learning_rate=0.007122273166129031,
                                  l2_regularization=8.931354194538156e-05,
                                  n_disc=3,
                                  generated_mse_imbalance=0.23927614146670084,
                                  generated_loss_imbalance=0.03568210662431517,
                                  kl_imbalance=0.00462105286455568,
                                  reconstruction_mse_imbalance=0.003925185127256372,
                                  likelihood_imbalance=2.3046966638597164)
        mse_all.append(mse)
        r_value_all.append(r_value)
        mae_all.append(mae)
        print("epoch---{}---r_value_ave  {}  mse_all_ave {}  mae_all_ave  {}  "
              "r_value_std {}----mse_all_std  {}  mae_std {}".
              format(i, np.mean(r_value_all), np.mean(mse_all), np.mean(mae_all),
                     np.std(r_value_all), np.std(mse_all), np.std(mae_all)))





























