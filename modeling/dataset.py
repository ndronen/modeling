import h5py
from sklearn.utils import check_random_state
import numpy as np
from modeling.utils import balanced_class_weights
from keras.utils import np_utils

class HDF5FileDataset(object):
    def __init__(self, file_path, data_name, target_name, batch_size, one_hot=True, random_state=17):
        assert isinstance(data_name, (list,tuple))
        assert isinstance(target_name, (list,tuple))

        random_state = check_random_state(random_state)

        self.__dict__.update(locals())
        del self.self

        self._load_data()
        self._check_data()

    def _load_data(self):
        self.hdf5_file = h5py.File(self.file_path)
        self.n_classes = {}
        for target_name in self.target_name:
            self.n_classes[target_name] = np.max(self.hdf5_file[target_name])+1

    def _check_data(self):
        self.n = None
        for data_name in self.data_name:
            if self.n is None:
                self.n = len(self.hdf5_file[data_name])
            else:
                assert len(self.hdf5_file[data_name]) == self.n
        for target_name in self.target_name:
            assert len(self.hdf5_file[target_name]) == self.n

    def __getitem__(self, name):
        return self.hdf5_file[name].value

    def class_weights(self, class_weight_exponent, target):
        return balanced_class_weights(
                self.hdf5_file[target],
                2,
                class_weight_exponent)

    def generator(self, one_hot=None, batch_size=None):
        if one_hot is None: one_hot = self.one_hot
        if batch_size is None: batch_size = self.batch_size

        while 1:
            idx = self.random_state.choice(self.n, size=batch_size, replace=False)
            batch = {}
            for data_name in self.data_name:
                batch[data_name] = self.hdf5_file[data_name].value[idx]
            for target_name in self.target_name:
                target = self.hdf5_file[target_name].value[idx]
                if one_hot:
                    batch[target_name] = np_utils.to_categorical(target,
                            self.n_classes[target_name])
                else:
                    batch[target_name] = target

            yield batch
