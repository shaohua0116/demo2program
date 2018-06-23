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
    num_train = config.num_train
    num_test = config.num_test
    num_val = config.num_val
    num_total = num_train + num_test + num_val

    # output files
    f = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    # progress bar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    dsl = get_KarelDSL(dsl_type='prob', seed=config.seed)
    s_gen = KarelStateGenerator(seed=config.seed)
    karel_world = karel.Karel_world()

    count = 0
    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    seen_programs = set()
    while(1):
        # generate a single program
        random_code = dsl.random_code(max_depth=config.max_program_stmt_depth,
                                      max_nesting_depth=config.max_program_nesting_depth)
        # skip seen programs
        if random_code in seen_programs:
            continue
        program_seq = np.array(dsl.code2intseq(random_code), dtype=np.int8)
        if program_seq.shape[0] > config.max_program_length:
            continue

        s_h_list = []
        a_h_list = []
        num_demo = 0
        num_trial = 0
        while num_demo < config.num_demo_per_program and \
                num_trial < config.max_demo_generation_trial:
            try:
                s, _, _, _, _ = s_gen.generate_single_state(h, w, wall_prob)
                karel_world.set_new_state(s)
                s_h = dsl.run(karel_world, random_code)
            except RuntimeError:
                pass
            else:
                if len(karel_world.s_h) <= config.max_demo_length and \
                        len(karel_world.s_h) >= config.min_demo_length:
                    s_h_list.append(np.stack(karel_world.s_h, axis=0))
                    a_h_list.append(np.array(karel_world.a_h))
                    num_demo += 1

            num_trial += 1

        if num_demo < config.num_demo_per_program:
            continue

        len_s_h = np.array([s_h.shape[0] for s_h in s_h_list], dtype=np.int16)
        if np.max(len_s_h) < config.min_max_demo_length_for_program:
            continue

        demos_s_h = np.zeros([num_demo, np.max(len_s_h), h, w, c], dtype=bool)
        for i, s_h in enumerate(s_h_list):
            demos_s_h[i, :s_h.shape[0]] = s_h

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)

        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        max_demo_length_in_dataset = max(max_demo_length_in_dataset, np.max(len_s_h))
        max_program_length_in_dataset = max(max_program_length_in_dataset, program_seq.shape[0])

        # save the state
        id = 'no_{}_prog_len_{}_max_s_h_len_{}'.format(
            count, program_seq.shape[0], np.max(len_s_h))
        id_file.write(id+'\n')
        grp = f.create_group(id)
        grp['program'] = program_seq
        grp['s_h_len'] = len_s_h
        grp['a_h_len'] = len_a_h
        grp['s_h'] = demos_s_h
        grp['a_h'] = demos_a_h
        seen_programs.add(random_code)
        # progress bar
        count += 1
        if count % (num_total / 100) == 0:
            bar.update(count / (num_total / 100))
        if count >= num_total:
            grp = f.create_group('data_info')
            grp['max_demo_length'] = max_demo_length_in_dataset
            grp['dsl_type'] = 'prob'
            grp['max_program_length'] = max_program_length_in_dataset
            grp['num_program_tokens'] = len(dsl.int2token)
            grp['num_demo_per_program'] = config.num_demo_per_program
            grp['num_action_tokens'] = len(dsl.action_functions)
            grp['num_train'] = config.num_train
            grp['num_test'] = config.num_test
            grp['num_val'] = config.num_val
            bar.finish()
            f.close()
            id_file.close()
            log.info('Dataset generated under {} with {}'
                     ' samples ({} for training and {} for testing '
                     'and {} for val'.format(dir_name, num_total,
                                             num_train, num_test, num_val))
            return


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='karel_dataset')
    parser.add_argument('--height', type=int, default=8,
                        help='height of square grid world')
    parser.add_argument('--width', type=int, default=8,
                        help='width of square grid world')
    parser.add_argument('--num_train', type=int, default=25000, help='num train')
    parser.add_argument('--num_test', type=int, default=5000, help='num test')
    parser.add_argument('--num_val', type=int, default=5000, help='num val')
    parser.add_argument('--wall_prob', type=float, default=0.1,
                        help='probability of wall generation')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--max_program_length', type=int, default=50)
    parser.add_argument('--max_program_stmt_depth', type=int, default=6)
    parser.add_argument('--max_program_nesting_depth', type=int, default=4)
    parser.add_argument('--min_max_demo_length_for_program', type=int, default=2)
    parser.add_argument('--min_demo_length', type=int, default=8,
                        help='min demo length')
    parser.add_argument('--max_demo_length', type=int, default=20,
                        help='max demo length')
    parser.add_argument('--num_demo_per_program', type=int, default=10,
                        help='number of seen demonstrations')
    parser.add_argument('--max_demo_generation_trial', type=int, default=100)
    args = parser.parse_args()

    args.dir_name = os.path.join('datasets/', args.dir_name)
    check_path(args.dir_name)

    generator(args)

if __name__ == '__main__':
    main()
