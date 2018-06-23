import argparse
import editdistance
import h5py
import numpy as np
from cv2 import resize, INTER_AREA
from tqdm import tqdm
from vizdoom_world import Vizdoom_world
from dsl.dsl_hit_analysis import hit_count
from dsl.vocab import VizDoomDSLVocab


def downsize(img, h=80, w=80):
    image_resize = resize(img, (h, w), interpolation=INTER_AREA)
    return image_resize

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_file', type=str, default='result.hdf5', help=' ')
parser.add_argument('--data_file', type=str,
                    default='datasets/vizdoom_dataset/data.hdf5', help=' ')
args = parser.parse_args()

fr = h5py.File(args.result_file, 'r')
ft = h5py.File(args.data_file, 'r')

perception_type = ft['data_info']['perception_type'].value
vocab = VizDoomDSLVocab(perception_type=perception_type)
world = Vizdoom_world(config='asset/default.cfg',
                      perception_type=perception_type)
world.init_game()
id_dict = {}
execute_correct = []
sequence_match = []
edit_distances = []

num_test_demo = ft['data_info']['num_test_demo_per_program'].value
vizdoom_pos_keys = list(ft['data_info']['vizdoom_pos_keys'].value)
for i, id in enumerate(tqdm(fr.keys())):
    id_dict[id] = i
    prog_len = fr[id]['pred_program_len'].value
    program_tokens = np.argmax(fr[id]['pred_program'].value, axis=0)[:prog_len]
    program_tokens_str = ''.join([str(t) for t in program_tokens])
    program = vocab.intseq2str(program_tokens)
    gt_program_tokens = ft[id]['program'].value
    gt_program_tokens_str = ''.join([str(t) for t in gt_program_tokens])
    gt_program = vocab.intseq2str(gt_program_tokens)

    edit_dist = int(editdistance.eval(program_tokens_str, gt_program_tokens_str))
    edit_distances.append(edit_dist)

    sequence_match.append(program == gt_program)

    hit_exe, hit_compile_success = hit_count(program)
    if not hit_compile_success:
        execute_correct.append(False)
        continue

    test_s_h = ft[id]['test_s_h'].value
    test_s_h_len = ft[id]['test_s_h_len'].value
    init_pos = ft[id]['test_vizdoom_init_pos'].value
    init_pos_len = ft[id]['test_vizdoom_init_pos_len'].value
    is_correct = True
    for k in range(num_test_demo):
        init_dict = {}
        for p, key in enumerate(vizdoom_pos_keys):
            init_dict[key] = np.squeeze(
                init_pos[k, p][:init_pos_len[k, p]])
        world.new_episode(init_dict)
        hit, num_cal, success = hit_exe(world, 0)
        if not success or len(world.s_h) == 1:
            is_correct = False
            break
        if len(world.s_h) != test_s_h_len[k]:
            is_correct = False
            break
        small_s_h = []
        for s in world.s_h:
            small_s_h.append(downsize(s, 80, 80))
        small_s_h = np.stack(small_s_h, 0)
        if not np.all(test_s_h[k, :test_s_h_len[k]] == small_s_h):
            is_correct = False
            break
    execute_correct.append(is_correct)

execute_correct = np.array(execute_correct).astype(np.int32)
sequence_match = np.array(sequence_match).astype(np.int32)
edit_distances = np.array(edit_distances).astype(np.int32)
for d in range(20):
    seq_acc = np.clip((sequence_match + (edit_distances <= d).astype(np.int32)), 0, 1).mean()
    exe_acc = np.clip((execute_correct + (edit_distances <= d).astype(np.int32)), 0, 1).mean()
    print('edit distance: {}, seq_acc: {}, exe_acc: {}'.format(d, seq_acc, exe_acc))

fr.close()
ft.close()
