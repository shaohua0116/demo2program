from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import h5py
from karel_env.util import log


rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, dataset_path, name='default', num_k=10, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.num_k = num_k
        self.is_train = is_train

        filename = 'data.hdf5'
        file = osp.join(dataset_path, filename)
        log.info("Reading %s ...", file)

        self.data = h5py.File(file, 'r')
        self.dsl_type = self.data['data_info']['dsl_type'].value
        self.num_demo = int(self.data['data_info']['num_demo_per_program'].value)
        self.max_demo_len = int(self.data['data_info']['max_demo_length'].value)
        self.max_program_len = int(self.data['data_info']['max_program_length'].value)
        self.num_program_tokens = int(self.data['data_info']['num_program_tokens'].value)
        self.num_action_tokens = int(self.data['data_info']['num_action_tokens'].value)
        if 'env_type' in self.data['data_info']:
            self.env_type = self.data['data_info']['env_type'].value
        else: self.env_type = None
        log.info("Reading Done: %s", file)

    def get_data(self, id, order=None):
        # preprocessing and data augmentation

        # each data point consist of a program + k demo

        # dim: [one hot dim of program tokens, program len]
        program_tokens = self.data[id]['program'].value
        program = np.zeros([self.num_program_tokens, self.max_program_len], dtype=bool)
        program[:, :len(program_tokens)][program_tokens, np.arange(len(program_tokens))] = 1
        padded_program_tokens = np.zeros([self.max_program_len], dtype=program_tokens.dtype)
        padded_program_tokens[:len(program_tokens)] = program_tokens

        demo_data = self.data[id]['s_h'].value
        test_demo_data = self.data[id]['test_s_h'].value

        if 'p_v_h' in self.data[id]:
            per_data = self.data[id]['p_v_h'].value
            test_per_data = self.data[id]['test_p_v_h'].value
        else:
            per_data = self.data[id]['per'].value
            test_per_data = self.data[id]['test_per'].value

        sz = demo_data.shape
        demo = np.zeros([sz[0], self.max_demo_len, sz[2], sz[3], sz[4]], dtype=demo_data.dtype)
        demo[:, :sz[1], :, :, :] = demo_data
        sz = test_demo_data.shape
        test_demo = np.zeros([sz[0], self.max_demo_len, sz[2], sz[3], sz[4]], dtype=demo_data.dtype)
        test_demo[:, :sz[1], :, :, :] = test_demo_data
        # dim: [k, action_space, max len of demo - 1]
        action_history_tokens = self.data[id]['a_h'].value
        action_history = []
        for a_h_tokens in action_history_tokens:
            # num_action_tokens + 1 is <e> token which is required for detecting
            # the end of the sequence. Even though the original length of the
            # action history is max_demo_len - 1, we make it max_demo_len, by
            # including the last <e> token.
            a_h = np.zeros([self.max_demo_len, self.num_action_tokens + 1], dtype=bool)
            a_h[:len(a_h_tokens), :][np.arange(len(a_h_tokens)), a_h_tokens] = 1
            a_h[len(a_h_tokens), self.num_action_tokens] = 1  # <e>
            action_history.append(a_h)
        action_history = np.stack(action_history, axis=0)
        padded_action_history_tokens = np.argmax(action_history, axis=2)

        # dim: [test_k, action_space, max len of demo - 1]
        test_action_history_tokens = self.data[id]['test_a_h'].value
        test_action_history = []
        for test_a_h_tokens in test_action_history_tokens:
            # num_action_tokens + 1 is <e> token which is required for detecting
            # the end of the sequence. Even though the original length of the
            # action history is max_demo_len - 1, we make it max_demo_len, by
            # including the last <e> token.
            test_a_h = np.zeros([self.max_demo_len, self.num_action_tokens + 1], dtype=bool)
            test_a_h[:len(test_a_h_tokens), :][np.arange(len(test_a_h_tokens)), test_a_h_tokens] = 1
            test_a_h[len(test_a_h_tokens), self.num_action_tokens] = 1  # <e>
            test_action_history.append(test_a_h)
        test_action_history = np.stack(test_action_history, axis=0)
        padded_test_action_history_tokens = np.argmax(test_action_history, axis=2)

        # program length: [1]
        program_length = np.array([len(program_tokens)], dtype=np.float32)

        # len of each demo. dim: [k]
        demo_length = self.data[id]['s_h_len'].value
        test_demo_length = self.data[id]['test_s_h_len'].value

        # per
        pad_per_data = np.pad(
            per_data, ((0, 0), (0, self.max_demo_len-per_data.shape[1]), (0, 0)),
            mode='constant', constant_values=0)
        pad_test_per_data = np.pad(
            test_per_data, ((0, 0), (0, self.max_demo_len-test_per_data.shape[1]), (0, 0)),
            mode='constant', constant_values=0)

        return program, padded_program_tokens, demo[:self.num_k], test_demo, \
            action_history[:self.num_k], padded_action_history_tokens[:self.num_k], \
            test_action_history, padded_test_action_history_tokens, \
            program_length, demo_length[:self.num_k], test_demo_length, \
            pad_per_data[:self.num_k], pad_test_per_data

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(dataset_path, num_k=10, is_train=True):
    ids_train, ids_test, ids_val = all_ids(dataset_path)

    dataset_train = Dataset(ids_train, dataset_path, name='train',
                            num_k=num_k, is_train=is_train)
    dataset_test = Dataset(ids_test, dataset_path, name='test',
                            num_k=num_k, is_train=is_train)
    dataset_val = Dataset(ids_val, dataset_path, name='val',
                            num_k=num_k, is_train=is_train)
    return dataset_train, dataset_test, dataset_val


def all_ids(dataset_path):
    with h5py.File(osp.join(dataset_path, 'data.hdf5'), 'r') as f:
        num_train = int(f['data_info']['num_train'].value)
        num_test = int(f['data_info']['num_test'].value)
        num_val = int(f['data_info']['num_val'].value)

    with open(osp.join(dataset_path, 'id.txt'), 'r') as fp:
        ids_total = [s.strip() for s in fp.readlines() if s]

    ids_train = ids_total[:num_train]
    ids_test = ids_total[num_train: num_train + num_test]
    ids_val = ids_total[num_train + num_test: num_train + num_test + num_val]

    rs.shuffle(ids_train)
    rs.shuffle(ids_test)
    rs.shuffle(ids_val)

    return ids_train, ids_test, ids_val
