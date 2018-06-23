import argparse
import h5py
import os
import numpy as np
from karel import Karel_world


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_name', type=str, default='datasets/karel_dataset',
                    help=" ")
args = parser.parse_args()


def s2per(demo_data, demo_data_len):
    per_stack = []
    for i in range(demo_data.shape[0]):
        per_stack_t = []
        for j in range(demo_data.shape[1]):
            if j < demo_data_len[i]:
                s = demo_data[i, j]
                k = Karel_world(s)
                per = np.array([k.front_is_clear(), k.left_is_clear(),
                                k.right_is_clear(), k.marker_present(),
                                k.no_marker_present()])
            else:
                per = np.zeros([5])
            per_stack_t.append(per)
        per_stack.append(np.stack(per_stack_t))
    per_stack = np.stack(per_stack)
    return per_stack

# Your dataset path to data.hdf5
dataset_path_all = [os.path.join(args.dir_name, 'data.hdf5')]
for dataset_path in dataset_path_all:
    f = h5py.File(dataset_path, 'r+')
    count = 0
    for key in f.keys():
        count += 1
        print('{}: {}/{}'.format(dataset_path, count-1, len(f.keys())-1))
        if not key == 'data_info':
            per = s2per(f[key]['s_h'], f[key]['s_h_len'])
            try:
                f.__delitem__(key+'/per')
            except:
                pass
            f[key]['per'] = per

            try:
                per = s2per(f[key]['test_s_h'], f[key]['test_s_h_len'])
                try:
                    f.__delitem__(key+'/test_per')
                except:
                    pass
                f[key]['test_per'] = per
            except:
                pass
    f.close()
