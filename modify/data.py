import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
from collections import defaultdict
from sklearn.preprocessing import normalize
import tensorflow as tf
import utils


class DataSet(object):
    def __init__(self, dynamic_features, batch_size):
        self._dynamic_features = dynamic_features
        self._num_examples = dynamic_features.shape[0]
        self._epoch_completed = 0
        self._batch_completed = 0
        self._index_in_epoch = 0
        self._batch_size = batch_size
        self._total_batches_no = int(self._num_examples /
                                           self._batch_size)

    def next_batch(self):
        '''
        Generator that produces the next batch untill epoch is completed.
        :return:
        '''
        for batch in range(self._total_batches_no):
            if self._batch_completed == 0:
                self._shuffle()
            start = self._index_in_epoch

            self._index_in_epoch += self._batch_size
            end = self._index_in_epoch
            yield self._dynamic_features[start:end, :, :]
            self._batch_completed += 1

        self._epoch_completed += 1
        self._batch_completed = 0
        self._index_in_epoch = 0
        return

    #@utils.with_reproducible_rng
    def _shuffle(self):
        # This is so confusing! If the input is ndarray this works, but later on
        # code complains that this is not a tensor (literally the next line
        # of code); if this is tensor, then the indexing does not work... No
        # offence, these are my comments to run through the code fast.
        index = np.arange(self._num_examples)
        # This shuffles in place:
        np.random.shuffle(index)
        #self._dynamic_features = self._dynamic_features[index]
        # So I'm using this method I do not know if this is as fast as
        # handling it as ndarray right before training, but I want to get
        # proficient in tensorflow, so there is that:

        self._dynamic_features = \
            tf.gather(
                self._dynamic_features,
                index
            )

    @property
    def dynamic_features(self):
        return self._dynamic_features

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @property
    def batch_completed(self):
        return self._batch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


def read_data():
    dynamic_features = np.load("../Trajectory_generate/Result/6_visit_format_normal.npy")
    feature = dynamic_features.reshape(-1, 30)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_normalization = scaler.transform(feature)
    feature_normalization = feature_normalization.reshape(-1, 6, 30)
    return DataSet(feature_normalization)


def read_gaucoma_data():
    dynamic_features = np.load('patients_data_od.npy')[:, :, 2:].astype(np.float32)  # OD data
    feature = dynamic_features.reshape(-1, dynamic_features.shape[2])
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_normalization = scaler.transform(feature)
    feature_normalization = feature_normalization.reshape(-1, 6, dynamic_features.shape[2])
    feature_normalization = np.concatenate((np.load('patients_data_od.npy')[:, :, 1].reshape(-1, 6, 1).astype(np.float32), feature_normalization), axis=2)
    return DataSet(feature_normalization)


if __name__ == '__main__':
    read_gaucoma_data()
    # data = read_data()
