from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import argparse
import numpy as np

from cv2 import resize, INTER_AREA
from tqdm import tqdm

from vizdoom_env import Vizdoom_env
from dsl.dsl_parse import parse as vizdoom_parse
from dsl.random_code_generator import DoomProgramGenerator
from dsl.vocab import VizDoomDSLVocab
from util import log


class DoomStateGenerator(object):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.x_max = 64
        self.x_min = -480
        self.y_max = 480
        self.y_min = 64

    def gen_rand_pos(self):
        return [self.rng.randint(self.x_min, self.x_max),
                self.rng.randint(self.y_min, self.y_max)]

    def get_pos_keys(self):
        return ['player_pos', 'demon_pos', 'hellknight_pos',
                'revenant_pos', 'ammo_pos']

    # generate an initial env
    def generate_initial_state(self, min_ammo=4, max_ammo=5,
                               min_monster=4, max_monster=5):
        """ h is y, w is x
            s = [{"player_pos": [x, y], "monster_pos": [[x1, y1], [x2, y2]]}]
        """
        s = {}
        locs = []
        s["player_pos"] = self.gen_rand_pos()
        s["demon_pos"] = []
        s["hellknight_pos"] = []
        s["revenant_pos"] = []
        s["ammo_pos"] = []
        locs.append(s["player_pos"])

        ammo_count = self.rng.randint(min_ammo, max_ammo + 1)
        demon_count = self.rng.randint(min_monster, max_monster + 1)
        hellknight_count = self.rng.randint(min_monster, max_monster + 1)
        revenant_count = self.rng.randint(min_monster, max_monster + 1)
        while(revenant_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["revenant_pos"].append(new_pos)
                locs.append(new_pos)
                revenant_count -= 1

        while(hellknight_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["hellknight_pos"].append(new_pos)
                locs.append(new_pos)
                hellknight_count -= 1

        while(demon_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["demon_pos"].append(new_pos)
                locs.append(new_pos)
                demon_count -= 1

        while(ammo_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["ammo_pos"].append(new_pos)
                locs.append(new_pos)
                ammo_count -= 1
        return s


def downsize(img, h=80, w=80):
    image_resize = resize(img, (h, w), interpolation=INTER_AREA)
    return image_resize


def generator(config):
    dir_name = config.dir_name

    image_dir = os.path.join(dir_name, 'images')
    check_path(image_dir)

    num_train = config.num_train
    num_test = config.num_test
    num_val = config.num_val
    num_total = num_train + num_test + num_val

    # output files
    f = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    num_demo = config.num_demo_per_program + config.num_test_demo_per_program
    world_list = []
    log.info('Initializing {} vizdoom environments...'.format(num_demo))
    for _ in range(num_demo):
        log.info('[{}/{}]'.format(_, num_demo))
        world = Vizdoom_env(config="vizdoom_env/asset/default.cfg",
                            perception_type='simple')
        world.init_game()
        world_list.append(world)
    log.info('done')
    h = config.height
    w = config.width
    c = world_list[0].channel

    gen = DoomStateGenerator(seed=config.seed)
    prog_gen = DoomProgramGenerator(seed=config.seed)

    percepts = world_list[0].get_perception_vector_cond()
    vizdoom_vocab = VizDoomDSLVocab(
        perception_type='simple')

    count = 0
    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    pos_keys = gen.get_pos_keys()
    max_init_poslen = -1
    pbar = tqdm(total=num_total)
    while True:
        init_states = []
        for world in world_list:
            init_states.append(gen.generate_initial_state())
            world.new_episode(init_states[-1])

        program, gen_success = prog_gen.random_code(
            percepts, world_list[:config.num_demo_per_program])
        if not gen_success:
            continue
        if len(program.split()) > config.max_program_length:
            continue

        program_seq = np.array(vizdoom_vocab.str2intseq(program), dtype=np.int8)

        exe, compile_success = vizdoom_parse(program)
        if not compile_success:
            print('compile failure')
            print('program: {}'.format(program))
            raise RuntimeError('Program compile failure should not happen')

        all_success = True
        for k, world in enumerate(world_list[config.num_demo_per_program:]):
            idx = k + config.num_demo_per_program
            world.new_episode(init_states[idx])
            new_w, num_call, success = exe(world, 0)
            if not success or len(world.s_h) < config.min_demo_length \
                    or len(world.s_h) > config.max_demo_length:
                all_success = False
                break
        if not all_success: continue

        s_h_len_fail = False
        for world in world_list:
            if len(world.s_h) < config.min_demo_length or \
                    len(world.s_h) > config.max_demo_length:
                s_h_len_fail = True
        if s_h_len_fail: continue

        program_seq = np.array(vizdoom_vocab.str2intseq(program), dtype=np.int8)

        s_h_list = []
        a_h_list = []
        p_v_h_list = []
        for k, world in enumerate(world_list):
            s_h_list.append(np.stack(world.s_h, axis=0).copy())
            a_h_list.append(np.array(
                vizdoom_vocab.action_strlist2intseq(world.a_h)))
            p_v_h_list.append(np.stack(world.p_v_h, axis=0).copy())

        len_s_h = np.array([s_h.shape[0] for s_h in s_h_list], dtype=np.int16)

        demos_s_h = np.zeros([num_demo, np.max(len_s_h), h, w, c], dtype=np.int16)
        for i, s_h in enumerate(s_h_list):
            downsize_s_h = []
            for t, s in enumerate(s_h):
                if s.shape[0] != h or s.shape[1] != w:
                    s = downsize(s, h, w)
                downsize_s_h.append(s.copy())
            demos_s_h[i, :s_h.shape[0]] = np.stack(downsize_s_h, 0)

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)

        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        demos_p_v_h = np.zeros([num_demo, np.max(len_s_h), len(percepts)], dtype=np.bool)
        for i, p_v in enumerate(p_v_h_list):
            demos_p_v_h[i, :p_v.shape[0]] = p_v

        max_demo_length_in_dataset = max(
                max_demo_length_in_dataset, np.max(len_s_h))
        max_program_length_in_dataset = max(
                max_program_length_in_dataset, program_seq.shape[0])

        # save the state
        id = 'no_{}_prog_len_{}_max_s_h_len_{}'.format(
            count, program_seq.shape[0], np.max(len_s_h))
        id_file.write(id+'\n')

        # data: [# demo, # pos_key, max(# pos), 2]
        # len: [# demo, #pos_key]
        np_init_states = {}
        np_init_state_len = {}
        pos_key_maxlen = -1
        for k in pos_keys:
            np_init_states[k] = []
            np_init_state_len[k] = []
            for s in init_states:
                np_pos = np.array(s[k], dtype=np.int32)
                if np_pos.ndim == 1:
                    np_pos = np.expand_dims(np_pos, axis=0)
                np_init_states[k].append(np_pos)
                np_init_state_len[k].append(np_pos.shape[0])
                pos_key_maxlen = max(pos_key_maxlen, np_pos.shape[0])
        max_init_poslen = max(max_init_poslen, pos_key_maxlen)

        # 3rd dimension is 2 as they are positions
        np_merged_init_states = np.zeros([num_demo, len(pos_keys),
                                          pos_key_maxlen, 2],
                                         dtype=np.int32)
        merged_pos_len = []
        for p, key in enumerate(pos_keys):
            single_key_pos_len = []
            for k, state in enumerate(np_init_states[key]):
                np_merged_init_states[k, p, :state.shape[0]] = state
                single_key_pos_len.append(state.shape[0])
            merged_pos_len.append(np.array(single_key_pos_len, dtype=np.int32))
        np_merged_pos_len = np.stack(merged_pos_len, axis=1)

        grp = f.create_group(id)
        grp['program'] = program_seq
        grp['s_h_len'] = len_s_h[:config.num_demo_per_program]
        grp['s_h'] = demos_s_h[:config.num_demo_per_program]
        grp['a_h_len'] = len_a_h[:config.num_demo_per_program]
        grp['a_h'] = demos_a_h[:config.num_demo_per_program]
        grp['p_v_h'] = demos_p_v_h[:config.num_demo_per_program]
        grp['test_s_h_len'] = len_s_h[config.num_demo_per_program:]
        grp['test_s_h'] = demos_s_h[config.num_demo_per_program:]
        grp['test_a_h_len'] = len_a_h[config.num_demo_per_program:]
        grp['test_a_h'] = demos_a_h[config.num_demo_per_program:]
        grp['test_p_v_h'] = demos_p_v_h[config.num_demo_per_program:]
        grp['vizdoom_init_pos'] = \
            np_merged_init_states[:config.num_demo_per_program]
        grp['vizdoom_init_pos_len'] = \
            np_merged_pos_len[:config.num_demo_per_program]
        grp['test_vizdoom_init_pos'] = \
            np_merged_init_states[config.num_demo_per_program:]
        grp['test_vizdoom_init_pos_len'] = \
            np_merged_pos_len[config.num_demo_per_program:]

        count += 1
        pbar.update(1)
        if count >= num_total:
            grp = f.create_group('data_info')
            grp['max_demo_length'] = max_demo_length_in_dataset
            grp['max_program_length'] = max_program_length_in_dataset
            grp['num_program_tokens'] = len(vizdoom_vocab.int2token)
            grp['num_demo_per_program'] = config.num_demo_per_program
            grp['num_test_demo_per_program'] = config.num_test_demo_per_program
            grp['num_action_tokens'] = len(vizdoom_vocab.action_int2token)
            grp['num_train'] = config.num_train
            grp['num_test'] = config.num_test
            grp['num_val'] = config.num_val
            grp['s_h_h'] = h
            grp['s_h_w'] = w
            grp['s_h_c'] = c
            grp['percepts'] = percepts
            grp['vizdoom_pos_keys'] = pos_keys
            grp['vizdoom_max_init_pos_len'] = max_init_poslen
            grp['perception_type'] = 'simple'
            f.close()
            id_file.close()
            print('Dataset generated under {} with {}'
                  ' samples ({} for training and {} for testing '
                  'and {} for val'.format(dir_name, num_total,
                                          num_train, num_test, num_val))
            pbar.close()
            return


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='vizdoom_small', help=' ')
    parser.add_argument('--num_train', type=int, default=10000, help=' ')
    parser.add_argument('--num_test', type=int, default=1000, help=' ')
    parser.add_argument('--num_val', type=int, default=100, help=' ')
    parser.add_argument('--seed', type=int, default=123, help=' ')
    parser.add_argument('--max_program_length', type=int, default=32)
    parser.add_argument('--min_demo_length', type=int, default=2)
    parser.add_argument('--max_demo_length', type=int, default=8, help=' ')
    parser.add_argument('--num_demo_per_program', type=int, default=40, help=' ')
    parser.add_argument('--num_test_demo_per_program', type=int, default=10, help=' ')
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--height', type=int, default=80)
    args = parser.parse_args()

    args.dir_name += '_len{}_seed{}'.format(
        args.max_demo_length, args.seed)

    args.dir_name = os.path.join('datasets/', args.dir_name)
    check_path('datasets')
    check_path(args.dir_name)

    generator(args)


if __name__ == '__main__':
    main()
