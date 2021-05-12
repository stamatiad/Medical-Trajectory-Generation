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
import skopt
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective
from skopt import dump, load
from skopt.callbacks import CheckpointSaver






class ProposedModel():
    def __init__(self, save_model=True):
        '''
        Loads and splits dataset into train/test
        '''

        self.epochs = 100
        self.fout_name = f"test2_e{self.epochs}"
        self.save_model = save_model
        self.k_outer = -1

        # Both the numbers below are not array indices.
        # This is the starting visit that the algorithm sees. I set it to 1,
        # since it is the first ever patient checkpoint/admission (month 0 in the
        # dataset).
        self.previous_visit = 1
        self.previous_visit_idx = self.previous_visit - 1
        # This is the future admissions that the algorithm will predict. I set it
        # to 3 since this is the available ones that I have and I want to
        # maximally utilize my data (corresponds to M3, M6 admissions).
        self.predicted_visit = 3
        self.predicted_visit_idx = self.predicted_visit - 1

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
        self.features = vals[valid_cols_idx]
        self.features.sort()
        #TODO: MANUALLY REMOVE the Date feature (they are nan):
        self.features = np.delete(self.features, 16)
        valid_cols = valid_cols[3:]

        total_features = len(self.features)
        patient_no = raw_data.shape[0]
        # make sure that the features across time points (months) are consistent!
        valid_cols.sort()
        data_tmp = raw_data[valid_cols].copy()

        # Handle the categorical features MANUALLY:
        self.categorical_features = self.features[:15]
        # Replace the features in the dataframe:
        for col in data_tmp.columns.values:
            # if column contains categorical data (feature):
            categorical = any(s for s in self.categorical_features if col[3:]
                              in s)
            if categorical:
                previous_vals = data_tmp[col].values
                vals, ordinal_vals = \
                    np.unique(previous_vals, return_inverse=True)
                replace_d = {**dict(zip(vals[ordinal_vals], ordinal_vals)),
                    **{np.nan: -1}}
                data_tmp[col].replace(replace_d, inplace=True)

        # Whiten the continuous features MANUALLY:
        # This belongs to data preprocessing! In its own function


        # Transform the data in order to have the time frame also. Dims are:
        # patients, features, timepoints
        data_mat_tmp = np.transpose(
            data_tmp.values.reshape(-1, 3, total_features),
            (0, 2, 1)
        )
        features_arr = self.features
        self.feature_dims = len(features_arr)

        # TEST!! Create easy dataset to test fitting:
        '''
        blah = np.ones(data_mat_tmp.shape, dtype=float)
        blah[:, :, 1] = 2
        blah[:, :, 2] = 3
        for feat in range(self.feature_dims):
            blah[:, feat, :] += 3 * feat
        data_mat_tmp = blah
        '''

        # Now to make them Dataframe, MultiIndex compatible you need:
        data_mat = data_mat_tmp.reshape(-1, 3)
        patients_d = {i: f"Patient_{i}" for i in range(patient_no)}
        # patients_arr = np.arange(0, len(raw_data))
        timepoints_d = {i: f"M_{i}" for i in range(0, 9, 3)}
        # timepoints_arr = np.arange(0, 3)

        midx = pd.MultiIndex.from_product([patients_d.values(), features_arr])
        # Finally create the multidimentional table (index in pandas) with all
        # the data. This makes it easy to handle. The testing/training functions
        # will also handle it well.
        self.data_all = pd.DataFrame(data_mat, index=midx,
                                columns=timepoints_d.values())

        # Fill the NaN values with a value that does not appear naturally in the
        # input to signify a missing value (Can I do this?).
        self.data_all.fillna(-1, inplace=True)

        # Divide to train/test set: (basic up to this phase):
        # Do this with sklearn:
        self.split = [0.8, 0.2, 0.1]
        #split_seed = 123
        # This requires df to be sorted, Why it is not?
        # blah = data_all.loc[(slice(None), slice('Apetite_QLQ30')), :]

        # Initialize defalut model parameters (from the original paper):
        self.hidden_size = 32
        self.z_dims = 64
        self.learning_rate = 0.007122273166129031
        self.l2_regularization = 8.931354194538156e-05
        self.kl_imbalance = 0.00462105286455568
        self.reconstruction_mse_imbalance = 0.003925185127256372
        self.generated_mse_imbalance = 0.23927614146670084
        self.likelihood_imbalance = 2.3046966638597164
        self.generated_loss_imbalance = 0.03568210662431517


        # Do the K CV split of the data:
        self.K = 3
        #TODO: seed
        seed = 0
        self.K_idx = {}
        # Do split the data based on seed:
        idx_all = np.random.permutation(patient_no)
        step = int(patient_no / self.K)
        for k in range(self.K):
            self.K_idx[k] = np.array(idx_all[k*step:(k+1)*step])



        # Try to visualize/white the data to get a grip:

        if False:
            for feature in features_arr:
                fig, ax = plt.subplots()
                blah = self.data_all.xs(feature, level=1, drop_level=False)
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
                # plt.show()
                plt.savefig(f'Feature_{feature}.png')
                plt.close()


    def preprocessing(self, train_idx=None, test_idx=None, inner=True):
        '''
        This should be called before each train (inner or outer). It
        normalizes all the data with mu/sigma estimated from train data only!
        :param data: the initial chunk of data (from inner or outer CV loop)
        :return:
        '''

        patient_idx = self.data_all.index.unique(level=0)
        # Save the dataframes to the correct locations, so the model knows
        # what to do:
        self.train_df = self.data_all.loc[patient_idx[train_idx], :, :].copy()
        if inner:
            # If on inner nested CV loop:
            self.valid_df = \
                self.data_all.loc[patient_idx[test_idx], :, :].copy()
        else:
            # If on outer nested CV loop:
            self.test_df = \
                self.data_all.loc[patient_idx[test_idx], :, :].copy()


        # Normalize the continuous features MANUALLY:
        self.continuous_features_d = {k:(0,0) for k in self.features[15:]}
        pdidx = pd.IndexSlice


        for feature in self.continuous_features_d.keys():
            # isolate features on the df:
            # Get normalization ONLY FROM TRAIN; we are not supposed to peek
            # into validation/test!!!
            arr_tmp = self.train_df.loc[:, feature, :].values
            # Do not count missing values for the normalization:
            arr_tmp[arr_tmp == -1.0] = np.nan
            mu = np.nanmean(arr_tmp)
            std = np.nanstd(arr_tmp)
            # save mu, std for use in validation/testing:
            self.continuous_features_d[feature] = (mu, std)
            # Normalize the data:
            self.train_df.loc[pdidx[:, feature, :], :] = \
                (self.train_df.loc[pdidx[:, feature, :], :] - mu) / std
            if inner:
                # If on inner nested CV loop:
                self.valid_df.loc[pdidx[:, feature, :], :] = \
                    (self.valid_df.loc[pdidx[:, feature, :], :] - mu) / std
            else:
                # If on outer nested CV loop:
                self.test_df.loc[pdidx[:, feature, :], :] = \
                    (self.test_df.loc[pdidx[:, feature, :], :] - mu) / std




    def unstack_to_mat(self, df, nfeatures):
        # unstack getting the dims from args, because df is giving me hard time.
        m = int(df.shape[0] / nfeatures)
        n = int(nfeatures)
        # m, n = len(df.index.levels[-1]), len(df.index.levels[1])
        arr = df.values.reshape(m, n, -1).swapaxes(1, 2)
        return arr


    #@tf.function
    def train(self, save_model=False, run_valid=True, **kwargs):

        # get training params in case we are calling train() from the optimizer:
        # This SILENTLY will replace non given optimizer parameters with
        # default ones. I am not sure I want this as the default behaviour.
        hidden_size = kwargs.get('hidden_size', self.hidden_size)
        z_dims = kwargs.get('z_dims', self.z_dims)
        l2_regularization = kwargs.get('l2_regularization', self.l2_regularization)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        generated_mse_imbalance = kwargs.get('generated_mse_imbalance', self.generated_mse_imbalance)
        generated_loss_imbalance = kwargs.get('generated_loss_imbalance', self.generated_loss_imbalance)
        kl_imbalance = kwargs.get('kl_imbalance', self.kl_imbalance)
        reconstruction_mse_imbalance = kwargs.get('reconstruction_mse_imbalance', self.reconstruction_mse_imbalance)
        likelihood_imbalance = kwargs.get('likelihood_imbalance', self.likelihood_imbalance)

        batch_size = 256 #64
        epochs = self.epochs

        #m, n = len(df.index.levels[-1]), len(df.index.levels[1])

        # Cast data to tensors:
        train_mat_tensor = tf.constant(
            self.unstack_to_mat(self.train_df, self.feature_dims),
            dtype=tf.float16
        )
        valid_mat_tensor = tf.constant(
            self.unstack_to_mat(self.valid_df, self.feature_dims),
            dtype=tf.float16
        )

        train_set = DataSet(train_mat_tensor, batch_size=batch_size)
        train_set.epoch_completed = 0
        valid_set = DataSet(valid_mat_tensor, batch_size=batch_size)
        valid_set.epoch_completed = 0


        # Define debug global vars:
        t_len = epochs * 12
        epochs_loss_arr = []#np.zeros((1, t_len), dtype=float)
        #epochs_loss_arr = tf.TensorArray(dtype=tf.int32, size=0,
        # dynamic_size=True)

        #mae_generated_arr = np.zeros((1, t_len), dtype=float)
        epochs_val_loss_arr = []#np.zeros((1, t_len), dtype=float)
        #mae_generated_valid_arr = np.zeros((1, t_len), dtype=float)
        train_iter = 0

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

        print(f'feature_dims---{self.feature_dims}')

        print(f'previous_visit---{self.previous_visit}---predicted_visit----'
              f'{self.predicted_visit}-')

        print(f'hidden_size---{hidden_size}---z_dims---'
              f'{z_dims}---l2_regularization---{z_dims}'
              f'---learning_rate---{learning_rate}--'
              f'generated_mse_imbalance---'
              f'{generated_mse_imbalance}---generated_loss_imbalance---'
              f'{generated_loss_imbalance}---'
              f'kl_imbalance---{kl_imbalance}---reconstruction_mse_imbalance---'
              f'{reconstruction_mse_imbalance}---'
              f'likelihood_imbalance---{likelihood_imbalance}')

        encode_share = Encoder(
            hidden_size=hidden_size
        )
        decoder_share = Decoder(
            hidden_size=hidden_size,
            feature_dims=self.feature_dims
        )

        post_net = Post(
            z_dims=z_dims
        )
        prior_net = Prior(
            z_dims=z_dims
        )

        #hawkes_process = HawkesProcess()


        count = 0
        optimizer_generation = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )
        optimizer_discriminator = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )
        cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )

        logged = set()
        max_loss = 0.001
        max_pace = 0.0001


        for epoch in range(epochs):
            epoch_loss = 0
            for train_batch in train_set.next_batch():
                batch_loss = 0

                # It the second dim is the time, what is this?? Should be swapped; Also
                # from the usage of var generated_trajectory I also deduce that the
                # second dim is the time axis!!!
                # After reconsideration the second dim is the features (makes sense
                # to me). Don't sweat it; do it again.

                #input_x_train = tf.cast(input_train[:, 1:, :], tf.float32)
                # This input_t_train should be the times of admissions. Should be:
                # batch_size x times of admissions.
                #input_t_train = tf.cast(input_train[:, 0, :], tf.float32)
                #TODO: make sure it works after changes in prev/pred:
                input_t_train = tf.constant(
                    np.repeat(
                        np.arange(
                            self.previous_visit_idx,
                            (self.previous_visit_idx + self.predicted_visit_idx)
                            * 3 +1 ,
                            3
                        ),
                        train_batch.shape[0]
                    ).reshape(
                    train_batch.shape[0], 3, order='F'),
                    dtype=tf.float16
                )
                batch = train_batch.shape[0]

                with tf.GradientTape() as gen_tape:
                    generated_trajectory = tf.zeros(shape=[batch, 0,
                                                           self.feature_dims])
                    #probability_likelihood = tf.zeros(shape=[batch, 0, 1])
                    reconstructed_trajectory = tf.zeros(shape=[batch, 0,
                                                               self.feature_dims])
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
                    for input_t, input_t_1, t, t_1 in utils.generate_inputs(
                        input=train_batch,
                        previous=self.previous_visit,
                        predicted=self.predicted_visit
                    ):
                        # t is the current time (patient visit vector data). t_idx is
                        # the array index, so t_idx = t-1 .

                    #for predicted_visit_ in range(1, predicted_visit):
                        # Also struggling to understand:
                        # Can I write this as an single feed to LSTM?

                        # previous_visit_ will always be 0 at the beginning of the loop:
                        encode_c = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        encode_h = tf.Variable(tf.zeros(shape=[batch, hidden_size]))
                        for t_idx in range(t):
                            # why we need to feed each one of the time slices and hot
                            # the whole sequence?? Does not the code handles the
                            # whole sequence?
                            encode_c, encode_h = \
                                encode_share(
                                    [train_batch[:, t_idx, :], encode_c, encode_h]
                                )

                        encode_h_i = encode_h  # h_i
                        encode_c, encode_h_i_1 = \
                            encode_share([input_t_1, encode_c, encode_h_i]) #
                        # h_(i+1)


                        # Create Prior/Posterior networks:
                        z_post, z_mean_post, z_log_var_post = \
                            post_net(
                                [
                                    encode_h_i,
                                    encode_h_i_1
                                ]
                            )
                        z_prior, z_mean_prior, z_log_var_prior = \
                            prior_net(
                                encode_h_i
                            )

                        # Prospa8w na katalabw ti ginetai me to previous visit. Den
                        # einai apla to starting point? Edw blepw oti to
                        # current_time_index_shape einai megalytero apo to diff
                        # previous-predicted pou 8a perimena...
                        # As ftasw mexri edw k blepw meta..
                        # TODO: make the HP returning a uniform conditional intensity
                        #  function lambda(t). This is because we do not need the
                        #  extra modeling power of the HP, since our admission
                        #  intervals are uniformly spread checkpoints.

                        current_time_index_shape = \
                            tf.ones(shape=[t])

                        '''
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
                        '''
                        # TODO: for what time points are the generation and
                        #  reconstruction respectively?
                        generated_next_visit, decode_c_generate, decode_h_generate = \
                            decoder_share(
                                [
                                    z_prior,
                                    encode_h_i,
                                    input_t,
                                    decode_c_generate,
                                    decode_h_generate
                                ]
                            )

                        # reconstruction
                        reconstructed_next_visit, decode_c_reconstruction, \
                        decode_h_reconstruction = \
                            decoder_share(
                                [
                                    z_post,
                                    encode_h_i,
                                    input_t,
                                    decode_c_reconstruction,
                                    decode_h_reconstruction
                                ]
                            )

                        reconstructed_trajectory = \
                            tf.concat(
                                (
                                    reconstructed_trajectory,
                                    tf.reshape(reconstructed_next_visit,
                                               [batch, -1, self.feature_dims]
                                               )
                                ), axis=1
                            )

                        generated_trajectory = tf.concat(
                            (
                                generated_trajectory,
                                tf.reshape(generated_next_visit,
                                           [batch, -1, self.feature_dims]
                                           )
                            ), axis=1
                        )

                        z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, z_dims])), axis=1)
                        z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, z_dims])), axis=1)

                        z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, z_dims])), axis=1)
                        z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, z_dims])), axis=1)

                    # Disable discriminator as it does not provide much of
                    # performance (as by the authors). I can only use the MSE (Lr
                    # error in the paper).

                    #loss1 = tf.keras.losses.mean_squared_error(input_x_train[:,
                    # previous_visit:previous_visit + predicted_visit, :], generated_trajectory)

                    # Here I slice from the first generated vector (of time
                    # previous_visit) to the last avaliable. As the original authors,
                    # I leave out of MSE the first (t_0) vector since it will reduce
                    # the average error:
                    generated_mse_loss = tf.reduce_mean(
                        tf.keras.losses.mse(
                            train_batch[:,
                            self.previous_visit:self.previous_visit_idx +
                                                     self.predicted_visit, :],
                            generated_trajectory
                        )
                    )
                    reconstructed_mse_loss = tf.reduce_mean(
                        tf.keras.losses.mse(
                            train_batch[:,
                            self.previous_visit:self.previous_visit_idx +
                                                    self.predicted_visit, :],
                            reconstructed_trajectory
                        )
                    )

                    '''
                    std_post = tf.math.sqrt(tf.exp(z_log_var_post_all))
                    std_prior = tf.math.sqrt(tf.exp(z_log_var_prior_all))

                    kl_loss_element = 0.5 * (2 * tf.math.log(tf.maximum(std_prior, 1e-9)) - 2 * tf.math.log(
                        tf.maximum(std_post, 1e-9))
                                             + (tf.square(std_post) + (tf.square(z_mean_post_all - z_mean_prior_all)) /
                                                (tf.maximum(tf.square(std_prior), 1e-9))) - 1)
                    kl_loss = tf.reduce_mean(kl_loss_element)
                    '''

                    #likelihood_loss = tf.reduce_mean(probability_likelihood)

                    batch_loss = generated_mse_loss * generated_mse_imbalance +\
                            reconstructed_mse_loss * reconstruction_mse_imbalance
                    #+ \
                            #kl_loss * kl_imbalance #+ \
                            #likelihood_loss * likelihood_imbalance

                    variables = [var for var in encode_share.trainable_variables]
                    for weight in encode_share.trainable_variables:
                        batch_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

                    for weight in decoder_share.trainable_variables:
                        batch_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                        variables.append(weight)

                    for weight in post_net.trainable_variables:
                        batch_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                        variables.append(weight)

                    for weight in prior_net.trainable_variables:
                        batch_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                        variables.append(weight)

                    '''
                    for weight in hawkes_process.trainable_variables:
                        loss += tf.keras.regularizers.l2(l2_regularization)(weight)
                        variables.append(weight)
                    '''

                epoch_loss += batch_loss
                print(f"Batch completed {train_set.batch_completed}. Epoch "
                      f"completed {train_set.epoch_completed}. Batch loss:"
                      f" {batch_loss}. Epoch loss: {epoch_loss}.")

                # Perform backpropagation with batch_loss
                gradient_gen = gen_tape.gradient(batch_loss, variables)
                optimizer_generation.apply_gradients(zip(gradient_gen, variables))

                # Why does the author loads the weights? Should he be saving them
                # instead? Also the modulo operator is ALWAYS 0 so why is there?
                # Also authors keep only the first batch of each epoch logged???

                #if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed
                # not in logged:

                #TODO: I'm not sure what is this about:

                #encode_share.load_weights('encode_share_3_3_mimic.h5')
                #decoder_share.load_weights('decode_share_3_3_mimic.h5')
                #post_net.load_weights('post_net_3_3_mimic.h5')
                #prior_net.load_weights('prior_net_3_3_mimic.h5')
                #hawkes_process.load_weights('hawkes_process_3_3_mimic.h5')

                #logged.add(train_set.epoch_completed)
                loss_pre = generated_mse_loss

                mse_generated = tf.reduce_mean(
                    tf.keras.losses.mse(
                        train_batch[:,
                        self.previous_visit:self.previous_visit_idx
                                            +self.predicted_visit, :],
                        generated_trajectory
                    )
                )
                mae_generated = tf.reduce_mean(
                    tf.keras.losses.mae(
                        train_batch[:,
                        self.previous_visit:self.previous_visit_idx
                                        +self.predicted_visit, :],
                        generated_trajectory
                    )
                )

                #mse_generated_arr.append(mse_generated.numpy())
                #mae_generated_arr[0, train_iter] = mae_generated

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

                # ======================================================================
                # Training batch finishes
                # ======================================================================

            # ======================================================================
            # Training epoch finishes
            # ======================================================================

            # ======================================================================
            # Make validation to assess learning accuracy:
            # ======================================================================

            if run_valid:
                valid_loss = 0
                for valid_batch in valid_set.next_batch():
                    input_t_valid = tf.constant(
                        np.repeat(
                            np.arange(
                                self.previous_visit_idx,
                                (self.previous_visit_idx + self.predicted_visit_idx)
                                * 3 + 1, 3
                            ),
                            valid_batch.shape[0]
                        ).reshape(
                            valid_batch.shape[0], 3, order='F'),
                        dtype=tf.float16
                    )
                    batch_valid = valid_batch.shape[0]

                    # Test encode/prior/decode nets on prediction for time t+1 (t_1 in the
                    # code). Again use all the available data.
                    generated_trajectory_valid = tf.zeros(shape=[batch_valid, 0,
                                                                self.feature_dims])
                    for input_t, input_t_1, t, t_1 in utils.generate_inputs(
                            input=valid_batch,
                            previous=self.previous_visit,
                            predicted=self.predicted_visit
                    ):
                        encode_c_valid = \
                            tf.Variable(tf.zeros(shape=[batch_valid, hidden_size]))
                        encode_h_valid = \
                            tf.Variable(tf.zeros(shape=[batch_valid, hidden_size]))
                        for t_idx in range(t):
                            encode_c_valid, encode_h_valid = \
                                encode_share(
                                    [valid_batch[:, t_idx, :], encode_c_valid,
                                     encode_h_valid]
                                )
                        # h_i
                        context_state_valid = encode_h_valid


                        # Create Prior/Posterior networks:
                        z_prior_valid, z_mean_prior_valid, z_log_var_prior_valid = \
                            prior_net(
                                context_state_valid
                            )

                        # TODO: can I use something else than zero? Maybe some feature mean?
                        decode_c_generate_valid = tf.Variable(tf.zeros(shape=[batch_valid, hidden_size]))
                        decode_h_generate_valid = tf.Variable(tf.zeros(shape=[batch_valid, hidden_size]))

                        '''
                        current_time_index_shape_valid = \
                            tf.ones(shape=[t])
                        intensity_value_valid, likelihood_valid = \
                            hawkes_process([input_t_valid, current_time_index_shape_valid])
                        '''

                        # Generate the t+1 (t_1) patient vector:
                        input_t_1_generated, decode_c_generate_valid, \
                        decode_h_generate_valid = \
                            decoder_share(
                                [
                                    z_prior_valid,
                                    context_state_valid,
                                    input_t,
                                    decode_c_generate_valid,
                                    decode_h_generate_valid
                                ]
                            )

                        # Recreate the trajectory with the generated vector appended:
                        generated_trajectory_valid = tf.concat(
                            (
                                generated_trajectory_valid,
                                tf.reshape(
                                    input_t_1_generated,
                                    [batch_valid, -1, self.feature_dims]
                                )
                            ),
                            axis=1
                        )


                    mse_generated_valid = tf.reduce_mean(
                        tf.keras.losses.mse(
                            valid_batch[:,
                            self.previous_visit:self.previous_visit_idx
                                                +self.predicted_visit, :],
                            generated_trajectory_valid
                        )
                    )
                    mae_generated_valid = tf.reduce_mean(
                        tf.keras.losses.mae(
                            valid_batch[:,
                            self.previous_visit:self.previous_visit_idx
                                                +self.predicted_visit, :],
                            generated_trajectory_valid
                        )
                    )

                    #print(f"\t\tBATCHID = {train_set.batch_completed} MSE_TEST ="
                    #      f" {mse_generated_test}")
                    #print(f"\t\tBATCHID = {train_set.batch_completed} MAE_TEST ="
                    #      f" {mae_generated_test}")
                    #mae_generated_valid_arr[0, train_iter] = mae_generated_valid

                    # Gather validation across all batches:
                    valid_loss += mse_generated_valid

                print(f"\t\tValidation loss: {valid_loss}.")

                epochs_val_loss_arr.append(valid_loss.numpy() /
                                               valid_set._total_batches_no)

            # Epoch completed:
            epochs_loss_arr.append(epoch_loss.numpy() /
                                   train_set._total_batches_no)

        # Save the model to use it on test:
        '''
        if save_model:
            encode_share.save_weights(f'./checkpoints_{self.k_outer}/encoder')
            decoder_share.save_weights(f'./checkpoints_{self.k_outer}/decoder')
            post_net.save_weights(f'./checkpoints_{self.k_outer}/post_net')
            prior_net.save_weights(f'./checkpoints_{self.k_outer}/prior_net')
            #hawkes_process.save_weights(f'./checkpoints_{self.k_outer}/hawkes')

        tf.compat.v1.reset_default_graph()

        if run_valid:
            mse = mse_generated_valid
        else:
            mse = mse_generated
        '''
        mse = 0

        if True:
            if run_valid:
                fig, ax = plt.subplots()
                ax.plot(epochs_loss_arr, color='C0', label='MSE_train')
                ax.plot(epochs_val_loss_arr, color='C1', label='MSE_valid')
                plt.legend()
                plt.savefig(f'Error_e_{epochs}_{mse:.5f}.png')
                plt.close()

        sys.exit(0)
        return mse


    def test(self, **kwargs):


        # Params:
        hidden_size = kwargs.get('hidden_size', self.hidden_size)
        z_dims = kwargs.get('z_dims', self.z_dims)

        # Create the model:
        encode_share = Encoder(hidden_size=hidden_size)
        decoder_share = Decoder(
            hidden_size=hidden_size,
            feature_dims=self.feature_dims
        )
        prior_net = Prior(z_dims=z_dims)

        hawkes_process = HawkesProcess()


        input_x_test = tf.constant(
            self.unstack_to_mat(self.test_df, self.feature_dims),
            dtype=tf.float32
        )

        input_t_test = tf.constant(
            np.repeat(
                np.arange(
                    self.previous_visit_idx,
                    (self.previous_visit_idx + self.predicted_visit_idx) * 3
                    + 1,
                    3
                ),
                input_x_test.shape[0]
            ).reshape(
                input_x_test.shape[0], 3, order='F'),
            dtype=tf.float16
        )
        batch_test = input_x_test.shape[0]

        # Test encode/prior/decode nets on prediction for time t+1 (t_1 in the
        # code). Again use all the available data.
        generated_trajectory_test = tf.zeros(shape=[batch_test, 0, self.feature_dims],
                                dtype=tf.float32)
        # Use the first only timepoint to predict the rest HARDCODED MANUALLY:
        generated_trajectory_test = tf.concat(
            (
                generated_trajectory_test,
                tf.reshape(
                    input_x_test[:, 0, :],
                    [batch_test, -1, self.feature_dims]
                )
            ),
            axis=1
        )

        # load the trained weights:
        encode_share.load_weights(f'./checkpoints_{self.k_outer}/encoder')
        decoder_share.load_weights(f'./checkpoints_{self.k_outer}/decoder')
        prior_net.load_weights(f'./checkpoints_{self.k_outer}/prior_net')
        tmp = tf.ones(shape=[1])
        hawkes_process.build([input_t_test.shape, tmp.shape])
        hawkes_process.load_weights(f'./checkpoints_{self.k_outer}/hawkes')

        for t in range(1, self.predicted_visit):

            encode_c_test = \
                tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
            encode_h_test = \
                tf.Variable(tf.zeros(shape=[batch_test, hidden_size]))
            for t_idx in range(t):
                encode_c_test, encode_h_test = \
                    encode_share(
                        [generated_trajectory_test[:, t_idx, :], encode_c_test,
                         encode_h_test]
                    )

            # h_(i)
            context_state_test = encode_h_test

            # Create Prior/Posterior networks:
            z_prior_test, z_mean_prior_test, z_log_var_prior_test = \
                prior_net(
                    context_state_test
                )

            # TODO: can I use something else than zero? Maybe some feature mean?
            decode_c_generate_test = tf.Variable(
                tf.zeros(shape=[batch_test, hidden_size]))
            decode_h_generate_test = tf.Variable(
                tf.zeros(shape=[batch_test, hidden_size]))

            current_time_index_shape_test = \
                tf.ones(shape=[t])
            intensity_value_test, likelihood_test = \
                hawkes_process([input_t_test, current_time_index_shape_test])

            # Generate the t+1 (t_1) patient vector:
            input_t_1_generated, decode_c_generate_test, \
            decode_h_generate_test = \
                decoder_share(
                    [
                        z_prior_test,
                        context_state_test,
                        input_x_test[:, t, :],
                        decode_c_generate_test,
                        decode_h_generate_test * intensity_value_test
                    ]
                )

            # Recreate the trajectory with the generated vector appended:
            generated_trajectory_test = tf.concat(
                (
                    generated_trajectory_test,
                    tf.reshape(
                        input_t_1_generated,
                        [batch_test, -1, self.feature_dims]
                    )
                ),
                axis=1
            )

        mse_generated_test = tf.reduce_mean(
            tf.keras.losses.mse(
                input_x_test,
                generated_trajectory_test
            )
        )

        '''
        r_value_all = []
        p_value_all = []

        for r in range(predicted_visit):
            x_ = tf.reshape(input_x_test[:, previous_visit + r, :], (-1,))
            y_ = tf.reshape(generated_trajectory_test[:, r, :], (-1,))
            r_value_ = stats.pearsonr(x_, y_)
            r_value_all.append(r_value_[0])
            p_value_all.append(r_value_[1])
        '''

        return mse_generated_test #, np.mean(r_value_all)



    def evaluate(self, params):
        '''
        Can be used to evaluate a model externally.
        :return: The MSE score (for Bayesiann hyperparameter optimization to
        minimize).
        '''
        # The fact that params can not be a kwargs dict in modern python,
        # just, I have no words... Messy array expansion it is...

        # Expand the hyperparameters:
        hidden_size, z_dims, learning_rate, \
        l2_regularization, \
        kl_imbalance, \
        reconstruction_mse_imbalance, \
        generated_mse_imbalance, \
        likelihood_imbalance, \
        generated_loss_imbalance = params

        # make them a nice dict as they should be, then pass them around:
        params_d = {
            'hidden_size':hidden_size,
            'z_dims':z_dims,
            'learning_rate':learning_rate,
            'l2_regularization':l2_regularization,
            'generated_mse_imbalance':generated_mse_imbalance,
            'generated_loss_imbalance':generated_loss_imbalance,
            'kl_imbalance':kl_imbalance,
            'reconstruction_mse_imbalance':reconstruction_mse_imbalance,
            'likelihood_imbalance':likelihood_imbalance
        }

        # Use Keras to train the model.
        mse_all = []
        r_value_all = []
        mae_all = []

        print(f'hidden_size---{hidden_size}---z_dims---'
              f'{z_dims}---l2_regularization---{z_dims}'
              f'---learning_rate---{learning_rate}--'
              f'generated_mse_imbalance---'
              f'{generated_mse_imbalance}---generated_loss_imbalance---'
              f'{generated_loss_imbalance}---'
              f'kl_imbalance---{kl_imbalance}---reconstruction_mse_imbalance---'
              f'{reconstruction_mse_imbalance}---'
              f'likelihood_imbalance---{likelihood_imbalance}')

        # As a nested CV we need (for this hyperparameter setting) to
        # calculate the average score over all L (K-1 in our setting) folds.
        mse_arr = []
        for k_inner in range(self.K):
            if k_inner != self.k_outer:

                # Calculate inner CV indices:
                k_train_idx = np.zeros((0,), dtype=int)
                for k in range(self.K):
                    if k != self.k_outer and k != k_inner:
                        k_train_idx = np.concatenate(
                            (k_train_idx,self.K_idx[ k])
                        )
                k_test_idx = self.K_idx[k_inner]

                self.preprocessing(
                    train_idx=k_train_idx,
                    test_idx=k_test_idx,
                )
                mse = self.train(run_valid=True, **params_d)
                mse_arr.append(mse.numpy())

        tf.compat.v1.reset_default_graph()

        mse_average = np.array(mse_arr).mean()
        print(f"MSE:{mse_average}")
        return mse_average
        # This function exactly comes from :Hvass-Labs, TensorFlow-Tutorials

    def hyperparameter_optimization(self):
        '''
        Performs Bayesian hyperparameter optimization
        :return:
        '''
        dimensions = [
            hidden_size := Integer(low=4, high=64, name='hidden_size'),
            z_dims := Integer(low=4, high=64, name='z_dims'),
            learning_rate := Real(low=1e-6, high=1e-1, prior='log-uniform',
                                  name='learning_rate'),
            l2_regularization := Real(low=-5, high=1, name='l2_regularization'),
            kl_imbalance := Real(low=-6, high=1, name='kl_imbalance'),
            reconstruction_mse_imbalance := Real(low=-6, high=1,
                                                 name='reconstruction_mse_imbalance'),
            generated_mse_imbalance := Real(low=-6, high=1,
                                            name='generated_mse_imbalance'),
            likelihood_imbalance := Real(low=-6, high=3,
                                         name='likelihood_imbalance'),
            generated_loss_imbalance := Real(low=-6, high=1,
                                             name='generated_loss_imbalance'),

        ]
        dim_names = [
            "hidden_size",
            "z_dims",
            "learning_rate",
            "l2_regularization",
            "kl_imbalance",
            "reconstruction_mse_imbalance",
            "generated_mse_imbalance",
            "likelihood_imbalance",
            "generated_loss_imbalance",
        ]

        default_parameters = [32,
                              64,
                              0.007122273166129031,
                              8.931354194538156e-05,
                              0.00462105286455568,
                              0.003925185127256372,
                              0.23927614146670084,
                              2.3046966638597164,
                              0.03568210662431517,
                              ]

        x0 = default_parameters
        y0 = None

        checkpoint_saver = CheckpointSaver(f"k_{self.k_outer}_"
                                           f"e_{self.epochs}.pkl",
                                           compress=9)

        if False:
            search_result = load(f'Checkpoint_{self.fout_name}.pkl')
            x0 = search_result.x_iters
            y0 = search_result.func_vals

        #TODO: to seed the RNG, so hyperparams are the same across K folds!
        search_result = skopt.gp_minimize(
            func=self.evaluate,
            dimensions=dimensions,
            acq_func='EI',  # Expected Improvement.
            n_calls=11,
            x0=x0,
            y0=y0,
            callback=[checkpoint_saver],
            n_jobs=8
        )
        # dump results to continue from later on:
        # dump(search_result, 'search_results.pkl')

        if False:
            plot_convergence(search_result)
            #plt.show()
            plt.savefig(f'convergence_{self.fout_name}.png')
            plot_objective(result=search_result, dimensions=dim_names)
            #plt.show()
            plt.savefig(f'all_dims_{self.fout_name}.png', dpi=400)
            print('PRONTO!')

        # make them a nice dict as they should be, then pass them around:
        params_d = {
            'hidden_size': search_result.x[0],
            'z_dims': search_result.x[1],
            'learning_rate': search_result.x[2],
            'l2_regularization': search_result.x[3],
            'kl_imbalance': search_result.x[4],
            'reconstruction_mse_imbalance': search_result.x[5],
            'generated_mse_imbalance': search_result.x[6],
            'likelihood_imbalance': search_result.x[7],
            'generated_loss_imbalance': search_result.x[8],
        }

        # Return the best hyperparameters:
        return params_d, search_result



@utils.time_it
def main():
    warnings.filterwarnings(action='once')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #tf.debugging.set_log_device_placement(True)

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

    model = ProposedModel(save_model=False)

    # Do nested CV:
    k_fold_scores = []
    hyperparams_d = {}
    for k in range(model.K):
        model.k_outer = k

        # Calculate outer CV indices:
        k_train_idx = np.zeros((0,), dtype=int)
        for kk in range(model.K):
            if kk != model.k_outer:
                k_train_idx = np.concatenate(
                    (k_train_idx, model.K_idx[kk])
                )
        k_test_idx = model.K_idx[model.k_outer]

        model.preprocessing(
            train_idx=k_train_idx,
            test_idx=k_test_idx,
            inner=False
        )
        best_hyperparameters, skopt_result = model.hyperparameter_optimization()
        hyperparams_d[k]= (best_hyperparameters, skopt_result)

        # Compare best model /hypers to the test
        model.train(
            save_model=True,
            run_valid=False,
            **best_hyperparameters
        )
        mse_test = model.test(**best_hyperparameters)
        k_fold_scores.append(mse_test.numpy())

    # Average score over all K folds is the generalization error:
    gen_error = np.array(k_fold_scores).mean()

    print(f"Generalization ERROR: {gen_error}")

    # Train the model with the best hyperparameters
    #model.train(run_valid=False, **best_hyperparameters)

    if False:
        # Get optimal params
        #search_result = load(f'{model.fout_name}.pkl')
        search_result = load(f'checkpoint_test.pkl')
        params_d = {
            'hidden_size': search_result.x[0],
            'z_dims': search_result.x[1],
            'learning_rate': search_result.x[2],
            'l2_regularization': search_result.x[3],
            'kl_imbalance': search_result.x[4],
            'reconstruction_mse_imbalance': search_result.x[5],
            'generated_mse_imbalance': search_result.x[6],
            'likelihood_imbalance': search_result.x[7],
            'generated_loss_imbalance': search_result.x[8],
        }
        mse, mae, r_value = model.train(**params_d)

    print("Pronto!")


if __name__ == '__main__':
    print(f'Simulation is commencing (Bayesiean hyperparameter optimization)!')
    main()
    print(f'Simulation is over!')
