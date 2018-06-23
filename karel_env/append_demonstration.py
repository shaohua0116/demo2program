from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import argparse
import progressbar

import numpy as np

from dsl import get_KarelDSL
from util import log

import karel

""" Purpose of file is to append test demonstration data to existing dataset
"""


class KarelStateGenerator(object):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    # generate an initial env
    def generate_single_state(self, h=8, w=8, wall_prob=0.1):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[:, :, 4] = self.rng.rand(h, w) > 1 - wall_prob
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, h)
            x = self.rng.randint(0, w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now
        s[:, :, 6] = (self.rng.rand(h, w) > 0.9) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 5:]) == h*w, np.sum(s[:, :, :5])
        marker_weight = np.reshape(np.array(range(11)), (1, 1, 11))
        return s, y, x, np.sum(s[:, :, 4]), np.sum(marker_weight*s[:, :, 5:])


def generator(config):
    dir_name = config.dir_name
    h = config.height
    w = config.width
    c = len(karel.state_table)

    wall_prob = config.wall_prob

    # output files
    f = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'r+')
    dsl_type = f['data_info']['dsl_type'].value

    with open(os.path.join(dir_name, 'id.txt'), 'r') as id_file:
        ids = [s.strip() for s in id_file.readlines() if s]

    num_train = f['data_info']['num_train'].value
    num_test = f['data_info']['num_test'].value
    num_val = f['data_info']['num_val'].value
    num_total = num_train + num_test + num_val

    # progress bar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    dsl = get_KarelDSL(dsl_type=dsl_type, seed=config.seed)
    s_gen = KarelStateGenerator(seed=config.seed)
    karel_world = karel.Karel_world()

    count = 0
    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    for id_ in ids:
        grp = f[id_]
        # Reads a single program
        program_seq = grp['program'].value
        program_code = dsl.intseq2str(program_seq)

        test_s_h_list = []
        a_h_list = []
        num_demo = 0
        while num_demo < config.num_test_demo_per_program:
            try:
                s, _, _, _, _ = s_gen.generate_single_state(h, w, wall_prob)
                karel_world.set_new_state(s)
                s_h = dsl.run(karel_world, program_code)
            except RuntimeError:
                pass
            else:
                if len(karel_world.s_h) <= config.max_demo_length and \
                        len(karel_world.s_h) >= config.min_demo_length:
                    test_s_h_list.append(np.stack(karel_world.s_h, axis=0))
                    a_h_list.append(np.array(karel_world.a_h))
                    num_demo += 1

        len_test_s_h = np.array([s_h.shape[0] for s_h in test_s_h_list], dtype=np.int16)

        demos_test_s_h = np.zeros([num_demo, np.max(len_test_s_h), h, w, c], dtype=bool)
        for i, s_h in enumerate(test_s_h_list):
            demos_test_s_h[i, :s_h.shape[0]] = s_h

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)

        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        max_demo_length_in_dataset = max(max_demo_length_in_dataset, np.max(len_test_s_h))
        max_program_length_in_dataset = max(max_program_length_in_dataset, program_seq.shape[0])

        try:
            f.__delitem__(id_+'/test_s_h_len')
            f.__delitem__(id_+'/test_a_h_len')
            f.__delitem__(id_+'/test_s_h')
            f.__delitem__(id_+'/test_a_h')
        except:
            pass

        # Save testing state
        grp['test_s_h_len'] = len_test_s_h
        grp['test_a_h_len'] = len_a_h
        grp['test_s_h'] = demos_test_s_h
        grp['test_a_h'] = demos_a_h
        # progress bar
        count += 1
        if count % (num_total / 100) == 0:
            bar.update(count / (num_total / 100))

    try:
        f.__delitem__('data_info/num_test_demo_per_program')
    except:
        pass
    f['data_info']['num_test_demo_per_program'] = config.num_test_demo_per_program
    bar.finish()
    f.close()
    id_file.close()
    log.info('Dataset generated under {} with {}'
             ' samples ({} for training and {} for testing '
             'and {} for val'.format(dir_name, num_total,
                                     num_train, num_test, num_val))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='datasets/karel_dataset',
                        help=" ")
    parser.add_argument('--height', type=int, default=8,
                        help='height of square grid world')
    parser.add_argument('--width', type=int, default=8,
                        help='width of square grid world')
    parser.add_argument('--wall_prob', type=float, default=0.1,
                        help='probability of wall generation')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--min_max_demo_length_for_program', type=int, default=2)
    parser.add_argument('--min_demo_length', type=int, default=8,
                        help='min demo length')
    parser.add_argument('--max_demo_length', type=int, default=20,
                        help='max demo length')
    parser.add_argument('--num_test_demo_per_program', type=int, default=5,
                        help='number of unseen demonstrations')
    args = parser.parse_args()

    generator(args)

if __name__ == '__main__':
    main()
