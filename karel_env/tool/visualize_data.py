from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import argparse

from prompt_toolkit import prompt

from dsl import get_KarelDSL
from karel_util import state2symbol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='karel_default')
    args = parser.parse_args()

    dir_name = args.dir_name
    data_file = os.path.join(dir_name, 'data.hdf5')
    id_file = os.path.join(dir_name, 'id.txt')

    if not os.path.exists(data_file):
        print("data_file path doesn't exist: {}".format(data_file))
        return
    if not os.path.exists(id_file):
        print("id_file path doesn't exist: {}".format(id_file))
        return

    f = h5py.File(data_file, 'r')
    ids = open(id_file, 'r').read().splitlines()

    dsl = get_KarelDSL(seed=123)

    cur_id = 0
    while True:
        print('ids / previous id: {}'.format(cur_id))
        for i, id in enumerate(ids[max(cur_id - 5, 0):cur_id + 5]):
            print('#{}: {}'.format(max(cur_id - 5, 0) + i, id))

        print('Put id you want to examine')
        cur_id = int(prompt(u'In: '))

        print('code: {}'.format(dsl.intseq2str(f[ids[cur_id]]['program'])))
        print('demonstrations')
        for i, l in enumerate(f[ids[cur_id]]['s_h_len']):
            print('demo #{}: length {}'.format(i, l))
        print('Put demonstration number [0-{}]'.format(f[ids[cur_id]]['s_h'].shape[0]))
        demo_idx = int(prompt(u'In: '))
        seq_idx = 0

        print('code: {}'.format(dsl.intseq2str(f[ids[cur_id]]['program'])))
        state2symbol(f[ids[cur_id]]['s_h'][demo_idx][seq_idx])
        seq_idx += 1
        while seq_idx < f[ids[cur_id]]['s_h_len'][demo_idx]:
            print("Press 'c' to continue and 'n' to next example")
            print(seq_idx, f[ids[cur_id]]['s_h_len'][demo_idx])
            key = prompt(u'In: ')
            if key == 'c':
                print('code: {}'.format(dsl.intseq2str(f[ids[cur_id]]['program'])))
                state2symbol(f[ids[cur_id]]['s_h'][demo_idx][seq_idx])
                seq_idx += 1
            elif key == 'n':
                break
            else:
                print('Wrong key')
        print('Demo is terminated')


if __name__ == '__main__':
    main()
