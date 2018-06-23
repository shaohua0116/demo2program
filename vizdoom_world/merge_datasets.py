from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import argparse

from tqdm import tqdm


def merge(config):
    dir_name = os.path.join('datasets/', config.dir_name)
    check_path(dir_name)

    f = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    new_dataset_paths = list(set(config.dataset_paths))
    if len(new_dataset_paths) != len(config.dataset_paths):
        raise ValueError('There is overlap in the dataset paths')

    num_train, num_test, num_val = 0, 0, 0
    h, w, c = None, None, None
    max_demo_length = 0
    max_program_length = 0
    num_program_tokens = None
    num_demo_per_program = 0
    num_test_demo_per_program = 0
    num_action_tokens = None
    percepts = None
    vizdoom_pos_keys = None
    vizdoom_max_init_pos_len = 0
    perception_type = None
    print('data_info checking')
    for i, dataset_path in enumerate(config.dataset_paths):
        print('dataset [{}/{}]'.format(i, len(config.dataset_paths)))
        fs = h5py.File(os.path.join(dataset_path, 'data.hdf5'), 'r')
        fs_max_demo_length = fs['data_info']['max_demo_length'].value
        fs_max_program_length = fs['data_info']['max_program_length'].value
        fs_num_program_tokens = fs['data_info']['num_program_tokens'].value
        fs_num_demo_per_program = fs['data_info']['num_demo_per_program'].value
        fs_num_test_demo_per_program = fs['data_info']['num_test_demo_per_program'].value
        fs_num_action_tokens = fs['data_info']['num_action_tokens'].value
        fs_num_train = fs['data_info']['num_train'].value
        fs_num_test = fs['data_info']['num_test'].value
        fs_num_val = fs['data_info']['num_val'].value
        fs_h = fs['data_info']['s_h_h'].value
        fs_w = fs['data_info']['s_h_w'].value
        fs_c = fs['data_info']['s_h_c'].value
        fs_percepts = list(fs['data_info']['percepts'].value)
        fs_vizdoom_pos_keys = list(fs['data_info']['vizdoom_pos_keys'].value)
        fs_vizdoom_max_init_pos_len = fs['data_info']['vizdoom_max_init_pos_len'].value
        fs_perception_type = fs['data_info']['perception_type'].value

        max_demo_length = max(max_demo_length, fs_max_demo_length)
        max_program_length = max(max_program_length, fs_max_program_length)
        if num_program_tokens is None: num_program_tokens = fs_num_program_tokens
        elif num_program_tokens != fs_num_program_tokens:
            raise ValueError('program token mismatch: {}'.format(dataset_path))
        num_demo_per_program = max(num_demo_per_program, fs_num_demo_per_program)
        num_test_demo_per_program = max(num_test_demo_per_program,
                                        fs_num_test_demo_per_program)
        if num_action_tokens is None: num_action_tokens = fs_num_action_tokens
        elif num_action_tokens != fs_num_action_tokens:
            raise ValueError('num action token mismatch: {}'.format(dataset_path))
        num_train += fs_num_train
        num_test += fs_num_test
        num_val += fs_num_val
        if h is None: h = fs_h
        elif h != fs_h: raise ValueError('image height mismatch: {}'.format(dataset_path))
        if w is None: w = fs_w
        elif w != fs_w: raise ValueError('image width mismatch: {}'.format(dataset_path))
        if c is None: c = fs_c
        elif c != fs_c: raise ValueError('image channel mismatch: {}'.format(dataset_path))
        if percepts is None: percepts = fs_percepts
        elif percepts != fs_percepts:
            raise ValueError('percepts mismatch: {}'.format(dataset_path))
        if vizdoom_pos_keys is None: vizdoom_pos_keys = fs_vizdoom_pos_keys
        elif vizdoom_pos_keys != fs_vizdoom_pos_keys:
            raise ValueError('vizdoom_pos_keys mismatch: {}'.format(dataset_path))
        vizdoom_max_init_pos_len = max(vizdoom_max_init_pos_len, fs_vizdoom_max_init_pos_len)
        if perception_type is None: perception_type = fs_perception_type
        elif perception_type != fs_perception_type:
            raise ValueError('perception_type mismatch: {}'.format(dataset_path))
        fs.close()
    print('copy data')
    for i, dataset_path in enumerate(config.dataset_paths):
        print('dataset [{}/{}]'.format(i, len(config.dataset_paths)))
        fs = h5py.File(os.path.join(dataset_path, 'data.hdf5'), 'r')
        ids = open(os.path.join(dataset_path, 'id.txt'),
                   'r').read().splitlines()
        for id in tqdm(ids):
            new_id = '{}_{}'.format(i, id)

            id_file.write(new_id+'\n')
            grp = f.create_group(new_id)
            for key in fs[id].keys():
                grp[key] = fs[id][key].value
        fs.close()
    grp = f.create_group('data_info')
    grp['max_demo_length'] = max_demo_length
    grp['max_program_length'] = max_program_length
    grp['num_program_tokens'] = num_program_tokens
    grp['num_demo_per_program'] = num_demo_per_program
    grp['num_test_demo_per_program'] = num_test_demo_per_program
    grp['num_action_tokens'] = num_action_tokens
    grp['num_train'] = num_train
    grp['num_test'] = num_test
    grp['num_val'] = num_val
    grp['s_h_h'] = h
    grp['s_h_w'] = w
    grp['s_h_c'] = c
    grp['percepts'] = percepts
    grp['vizdoom_pos_keys'] = vizdoom_pos_keys
    grp['vizdoom_max_init_pos_len'] = vizdoom_max_init_pos_len
    grp['perception_type'] = perception_type
    f.close()
    id_file.close()
    print('Dataset generated under {} with {}'
          ' samples ({} for training and {} for testing '
          'and {} for val'.format(dir_name, num_train + num_test + num_val,
                                  num_train, num_test, num_val))


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise ValueError('Be careful, you are trying to overwrite some dir')


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='vizdoom_dataset')
    parser.add_argument('--dataset_paths', nargs='+',
                        help='list of existing dataset paths')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    merge(args)
