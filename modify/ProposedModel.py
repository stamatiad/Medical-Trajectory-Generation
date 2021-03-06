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

Debug = False

class ProposedModel():
    def __init__(self, save_model=True):
        ''' Loads and splits dataset into train/test
        '''

        self.epochs = 40
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
                    **{np.nan: np.nan}}
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
        # input to signify a missing value.
        # UPDATE: do not! You already replaced nan vals for categorical vars.
        # Let the continuous vars preprocessing handle these.
        #self.data_all.fillna(-1, inplace=True)

        # Divide to train/test set: (basic up to this phase):
        # Do this with sklearn:
        self.split = [0.8, 0.2, 0.1]
        #split_seed = 123
        # This requires df to be sorted, Why it is not?
        # blah = data_all.loc[(slice(None), slice('Apetite_QLQ30')), :]

        # Initialize defalut model parameters (from the original paper):
        self.hidden_size = 8 #32
        self.z_dims = 8 #64
        self.learning_rate = 0.007122273166129031
        self.l2_regularization = 8.931354194538156e-05
        #self.kl_imbalance = 0.00462105286455568
        self.reconstruction_mse_imbalance = 0.5
        self.generated_mse_imbalance = 0.5
        #self.reconstruction_mse_imbalance = 0.003925185127256372
        #self.generated_mse_imbalance = 0.23927614146670084
        #self.likelihood_imbalance = 2.3046966638597164
        #self.generated_loss_imbalance = 0.03568210662431517


        # Do the K CV split of the data:
        self.K = 3
        #TODO: seed
        seed = 0
        self.K_idx = {}
        # Do split the data based on seed:
        random_state = np.random.RandomState(0)
        idx_all = random_state.permutation(patient_no)
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


    def get_outer_fold_indices(self):
        # For all K folds:
        for k in range(self.K):
            #inform the state of the model
            self.k_outer = k
            # Calculate outer CV indices:
            k_train_idx = np.zeros((0,), dtype=int)
            for kk in range(self.K):
                if kk != self.k_outer:
                    k_train_idx = np.concatenate(
                        (k_train_idx, self.K_idx[kk])
                    )
            k_test_idx = self.K_idx[self.k_outer]
            yield (k, k_train_idx, k_test_idx)
        return

    def get_inner_fold_indices(self):
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
                yield (k_inner, k_train_idx, k_test_idx)
        return

    def preprocessing(self, train_idx=None, test_idx=None, inner=True):
        '''
        This should be called before each train (inner or outer). It
        normalizes all the data with mu/sigma estimated from train data only!
        :param data: the initial chunk of data (from inner or outer CV loop)
        :return:
        '''


        # DO NOT PEEK INTO THE FUTURE. First get only your assigned as
        # training data points.
        patient_idx = self.data_all.index.unique(level=0)

        train_df_tmp = self.data_all.loc[patient_idx[train_idx], :, :].copy()

        # =======================================================================
        # Missing values imputation.
        # =======================================================================

        # Get first level idx (different patients):
        # mi = self.data_all.index.levels[0]
        train_all_idx = train_df_tmp.index.unique(level=0)
        non_missing_idx = np.zeros((len(train_all_idx),), dtype=bool)
        for idx, patient in enumerate(train_all_idx):
            non_missing_idx[idx] = not train_df_tmp.loc[patient].isnull(
            ).values.any()

        self.train_df = train_df_tmp.loc[train_all_idx[non_missing_idx], :,
                        :].copy()

        # Get validation/test data points
        test_df_tmp = self.data_all.loc[patient_idx[test_idx], :, :].copy()

        # Remove patients with missing time points (who have missed entire
        # exam check points.
        test_all_idx = test_df_tmp.index.unique(level=0)
        non_missing_idx = np.zeros((len(test_all_idx),), dtype=bool)
        for idx, patient in enumerate(test_all_idx):
            tmp = test_df_tmp.loc[patient].isnull().values
            # Locate missing columns (checkpoints):
            non_missing_idx[idx] = np.logical_not(np.all(tmp, axis=0)).all()

        # Fill in any individual missing values using the kNN model:
        test_df_tmp = test_df_tmp.loc[test_all_idx[non_missing_idx]].copy()

        # TODO: I need a custom kNN that takes neighbors only from the same
        # checkpoint and not some future one.

        test_df_tmp.fillna(-1, inplace=True)


        # Save the dataframes to the correct locations, so the model knows
        # what to do:
        if inner:
            # If on inner nested CV loop:
            self.valid_df = \
                test_df_tmp.loc[test_all_idx[non_missing_idx], :, :].copy()
        else:
            # If on outer nested CV loop:
            self.test_df = \
                test_df_tmp.loc[test_all_idx[non_missing_idx], :, :].copy()


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

        # =======================================================================
        # Missing values imputation.
        # =======================================================================

        # Here I should use the train_df to create a kNN model and fill in
        # the missing values in validation/test data points.

        #print('done')




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
        reconstruction_mse_imbalance = kwargs.get('reconstruction_mse_imbalance', self.reconstruction_mse_imbalance)

        tf.compat.v1.reset_default_graph()

        batch_size = 64
        epochs = self.epochs

        #m, n = len(df.index.levels[-1]), len(df.index.levels[1])

        # Cast data to tensors:
        train_mat_tensor = tf.constant(
            self.unstack_to_mat(self.train_df, self.feature_dims),
            dtype=tf.float32
        )
        valid_mat_tensor = tf.constant(
            self.unstack_to_mat(self.valid_df, self.feature_dims),
            dtype=tf.float32
        )

        train_set = DataSet(train_mat_tensor, batch_size=batch_size)
        valid_set = DataSet(valid_mat_tensor, batch_size=batch_size)


        # Define debug global vars:
        epochs_loss_arr = tf.TensorArray(
            dtype=tf.float32, size=0,
            dynamic_size=True, clear_after_read=False)
        #mae_generated_arr = np.zeros((1, t_len), dtype=float)
        epochs_val_loss_arr = tf.TensorArray(
            dtype=tf.float32, size=0,
            dynamic_size=True, clear_after_read=False)
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


        print(f'hidden_size---{hidden_size}---'
              f'z_dims---{z_dims}---'
              f'l2_regularization---{l2_regularization}---'
              f'learning_rate---{learning_rate}--'
              f'generated_mse_imbalance---{generated_mse_imbalance}---'
              f'reconstruction_mse_imbalance---{reconstruction_mse_imbalance}---'
              )

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

        optimizer_generation = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

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

                batch = train_batch.shape[0]

                with tf.GradientTape() as gen_tape:
                    generated_trajectory = tf.zeros(shape=[batch, 0,
                                                           self.feature_dims])
                    #probability_likelihood = tf.zeros(shape=[batch, 0, 1])
                    reconstructed_trajectory = tf.zeros(shape=[batch, 0,
                                                               self.feature_dims])

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

                        '''
                        z_mean_post_all = tf.concat((z_mean_post_all, tf.reshape(z_mean_post, [batch, -1, z_dims])), axis=1)
                        z_mean_prior_all = tf.concat((z_mean_prior_all, tf.reshape(z_mean_prior, [batch, -1, z_dims])), axis=1)

                        z_log_var_post_all = tf.concat((z_log_var_post_all, tf.reshape(z_log_var_post, [batch, -1, z_dims])), axis=1)
                        z_log_var_prior_all = tf.concat((z_log_var_prior_all, tf.reshape(z_log_var_prior, [batch, -1, z_dims])), axis=1)
                        '''

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

                    # Keep track of loss without regularization (to be
                    # comparable with validation loss):

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

                epoch_loss += batch_loss

                if Debug:
                    print(f"Batch completed {train_set.batch_completed}. Epoch "
                          f"completed {train_set.epoch_completed}. Batch loss:"
                          f" {batch_loss}. Epoch loss: {epoch_loss}.")

                # Perform backpropagation with batch_loss
                gradient_gen = gen_tape.gradient(batch_loss, variables)
                optimizer_generation.apply_gradients(zip(gradient_gen, variables))

                #encode_share.load_weights('encode_share_3_3_mimic.h5')
                #decoder_share.load_weights('decode_share_3_3_mimic.h5')
                #post_net.load_weights('post_net_3_3_mimic.h5')
                #prior_net.load_weights('prior_net_3_3_mimic.h5')

                # ======================================================================
                # Training batch finishes
                # ======================================================================

            # ======================================================================
            # Training epoch finishes
            # ======================================================================
            # This is the training loss for a whole epoch.
            if Debug:
                print(f"\t\tTraining loss: "
                  f"{epoch_loss / train_set._total_batches_no}.")
            # Epoch completed:
            epochs_loss_arr = \
                epochs_loss_arr.write(epoch, epoch_loss / \
                                                    train_set._total_batches_no)

            # ======================================================================
            # Make validation to assess learning accuracy:
            # ======================================================================

            if run_valid:
                valid_loss = 0
                for valid_batch in valid_set.next_batch():
                    batch_valid = valid_batch.shape[0]

                    # Test encode/prior/decode nets on prediction for time t+1 (t_1 in the
                    # code). Again use all the available data.
                    generated_trajectory_valid = \
                        tf.zeros(shape=[batch_valid, 0, self.feature_dims],
                                 dtype=tf.float32)

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
                            self.previous_visit:self.previous_visit_idx +
                                                self.predicted_visit, :],
                            generated_trajectory_valid
                        )
                    )

                    # Gather validation across all batches:
                    valid_loss += mse_generated_valid

                # This is the validation loss for a whole epoch.
                if Debug:
                    print(f"\t\tValidation loss: "
                          f"{valid_loss / valid_set._total_batches_no}.")

                epochs_val_loss_arr = epochs_val_loss_arr.write(epoch,
                                                              valid_loss / \
                                               valid_set._total_batches_no)

            # ======================================================================
            # Validation epoch finishes
            # ======================================================================

        # Save the model to use it on test:
        if save_model:
            encode_share.save_weights(f'./checkpoints_{self.k_outer}/encoder')
            decoder_share.save_weights(f'./checkpoints_{self.k_outer}/decoder')
            post_net.save_weights(f'./checkpoints_{self.k_outer}/post_net')
            prior_net.save_weights(f'./checkpoints_{self.k_outer}/prior_net')

        if run_valid:
            loss = epochs_val_loss_arr.stack().numpy()[-1]
        else:
            loss = epochs_loss_arr.stack().numpy()[-1]


        return (loss, epochs_loss_arr.stack().numpy(),
        epochs_val_loss_arr.stack().numpy())


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

        input_x_test = tf.constant(
            self.unstack_to_mat(self.test_df, self.feature_dims),
            dtype=tf.float32
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

            # Generate the t+1 (t_1) patient vector:
            input_t_1_generated, decode_c_generate_test, \
            decode_h_generate_test = \
                decoder_share(
                    [
                        z_prior_test,
                        context_state_test,
                        input_x_test[:, t, :],
                        decode_c_generate_test,
                        decode_h_generate_test
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
        hidden_size, z_dims, \
        learning_rate, \
        l2_regularization = params
        #reconstruction_mse_imbalance, \
        #generated_mse_imbalance \

        # make them a nice dict as they should be, then pass them around:
        params_d = {
            'hidden_size':hidden_size,
            'z_dims':z_dims,
            'learning_rate':learning_rate,
            'l2_regularization':l2_regularization,
            #'reconstruction_mse_imbalance':reconstruction_mse_imbalance,
            #'generated_mse_imbalance':generated_mse_imbalance
        }

        # As a nested CV we need (for this hyperparameter setting) to
        # calculate the average score over all L (K-1 in our setting) folds.
        loss_arr = []
        loss_train_arr = []
        loss_valid_arr = []
        for k_inner, k_train_idx, k_test_idx in self.get_inner_fold_indices():
            self.preprocessing(
                train_idx=k_train_idx,
                test_idx=k_test_idx,
            )
            # This returns the validation loss:
            loss, train_loss, valid_loss = \
                self.train(run_valid=True, **params_d)
            loss_arr.append(loss)
            loss_train_arr.append(train_loss)
            loss_valid_arr.append(valid_loss)

        mse_average = np.array(loss_arr).mean()

        # Plot the average train/valid loss:
        if True:
            fig, ax = plt.subplots()
            ax.plot(np.array(loss_train_arr).mean(axis=0), color='C0',
                    label='MSE_train')
            ax.plot(np.array(loss_valid_arr).mean(axis=0), color='C1',
                    label='MSE_valid')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Average MSE')
            plt.legend()
            plt.savefig(f'Error_e_{self.epochs}_train_fold_'
                        f'{self.k_outer}_'
                        f'hd_{hidden_size}_'
                        f'zd_{z_dims}_'
                        f'lr{learning_rate}_'
                        f'l2_{l2_regularization}_'
                        f'mse{mse_average}.png')
            plt.close()

        return mse_average
        # This function exactly comes from :Hvass-Labs, TensorFlow-Tutorials

    def hyperparameter_optimization(self):
        '''
        Performs Bayesian hyperparameter optimization
        :return:
        '''
        dimensions = [
            hidden_size := Categorical([4, 8, 16], name='hidden_size'),
            z_dims := Categorical([4, 8, 16], name='z_dims'),
            learning_rate := Categorical([0.001, 0.005, 0.01, 0.05, 0.1],
                        name='learning_rate'),
            l2_regularization := Categorical([1.0e-2, 1.0e-3, 1.0e-4],
                        name='l2_regularization'),
        ]
        dim_names = [
            "hidden_size",
            "z_dims",
            "learning_rate",
            "l2_regularization",
            #"reconstruction_mse_imbalance",
            #"generated_mse_imbalance",
        ]

        default_parameters = [
            4,
            4,
            0.005,
            0.001,
                              ]

        x0 = default_parameters
        y0 = None

        checkpoint_saver = CheckpointSaver(f"CHECKPOINT.pkl",
                                           compress=9)
        search_result = load(f'CHECKPOINT.pkl')
        plot_convergence(search_result)
        # plt.show()
        plt.savefig(f'convergence_CHECKPOINT.png')
        plot_objective(result=search_result, dimensions=dim_names)
        # plt.show()
        plt.savefig(f'CHECKPOINT_GP.png', dpi=400)
        x0 = search_result.x_iters
        y0 = search_result.func_vals

        #TODO: to seed the RNG, so hyperparams are the same across K folds!
        search_result = skopt.gp_minimize(
            func=self.evaluate,
            dimensions=dimensions,
            acq_func='EI',  # Expected Improvement.
            n_calls=40,
            x0=x0,
            y0=y0,
            random_state=np.random.RandomState(0),
            callback=[checkpoint_saver],
            n_jobs=8
        )

        # make them a nice dict as they should be, then pass them around:
        params_d = {
            'hidden_size': search_result.x[0],
            'z_dims': search_result.x[1],
            'learning_rate': search_result.x[2],
            'l2_regularization': search_result.x[3],
            #'reconstruction_mse_imbalance': search_result.x[4],
            #'generated_mse_imbalance': search_result.x[5],
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
    # Loop the outer CV folds:
    for k, k_train_idx, k_test_idx in model.get_outer_fold_indices():

        # Loop the inner CV folds:
        best_hyperparameters, skopt_result = model.hyperparameter_optimization()
        hyperparams_d[k]= (best_hyperparameters, skopt_result)
        print(f"Best hyperparams of inner folds are: {best_hyperparameters}")

        model.preprocessing(
            train_idx=k_train_idx,
            test_idx=k_test_idx,
            inner=False
        )
        # Compare best model /hyperparameters to the test. Save the model for
        # the test to load it.
        model.train(
            save_model=True,
            run_valid=False,
            **best_hyperparameters
        )
        mse_test = model.test(**best_hyperparameters)
        k_fold_scores.append(mse_test.numpy())
        print(f"AVERAGE TEST LOSS (test fold: {k})"
              f":{mse_test}\n")

    # Average score over all K folds is the generalization error:
    gen_error = np.array(k_fold_scores).mean()

    print(f"Generalization ERROR: {gen_error}")

    # Train the model with the best hyperparameters:
    best_loss = np.inf
    best_hypers = None
    for (hyper, result) in hyperparams_d.values():
        if result.fun < best_loss:
            best_loss = result.fun
            best_hypers = hyper

    print(f"Best hyperparams of nested CV are: {best_hypers}")
    # Now the production model can be trained on all the existing data:
    #model.train(run_valid=False, **best_hyperparameters)


if __name__ == '__main__':
    print(f'Simulation is commencing (Bayesiean hyperparameter optimization)!')
    main()
    print(f'Simulation is over!')
