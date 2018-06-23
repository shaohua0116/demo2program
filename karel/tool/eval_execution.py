"""
Eval Execution

Execute output programs and then check execution accuracy and syntax accuracy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import collections
import h5py
import os
import numpy as np
from tqdm import tqdm

import karel
from dsl import get_KarelDSL
from dsl.dsl_parse import parse


def GetArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_hdf5', type=str)
    parser.add_argument('--output_hdf5', type=str)
    parser.add_argument('--output_log_path', type=str, default=None)
    parser.add_argument('--new_hdf5_path', type=str, default=None)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--dump', action='store_true', default=False)
    return parser.parse_args()


class CheckProgramOutput(
    collections.namedtuple("CheckProgramOutput",
                           ("data_id", "program", "syntax", "num_correct", "demo_correctness"))):
    pass


def CheckProgram(program, data_id, num_demo, demo, demo_len, dsl, karel_world):
    exe, s_exe = parse(program)
    if not s_exe:
        syntax = False
        demo_correctness = np.array([False] * num_demo)
        num_correct = 0
    else:
        syntax = True
        demo_correctness = np.array([False] * num_demo)
        for k in range(num_demo):
            init_state = demo[k][0]
            karel_world.set_new_state(init_state)
            karel_world.clear_history()
            exe, s_exe = parse(program)
            if not s_exe:
                raise RuntimeError('This should be correct')
            karel_world, n, s_run = exe(karel_world, 0)
            if not s_run:
                demo_correctness[k] = False
            else:
                exe_result_len = len(karel_world.s_h)
                exe_result = np.stack(karel_world.s_h)
                demo_correctness[k] = (demo_len[k] == exe_result_len and
                                       np.all(demo[k][:demo_len[k]] == exe_result))
        num_correct = demo_correctness.astype(np.int32).sum()
    return CheckProgramOutput(data_id, program, syntax, num_correct, demo_correctness)


class EvaluationResult:

    def __init__(self, name):
        self.name = name
        self.initialize()

    def initialize(self):
        self.syntax = []
        self.num_correct_count = {}
        self.demo_correctness = []
        self.ids = []
        self.programs = []

    def get_program_by_id(self, id):
        idx = self.ids.index(id)
        return self.programs[idx]

    def get_demo_correctness_by_id(self, id):
        idx = self.ids.index(id)
        return self.demo_correctness[idx]

    def get_syntax_by_id(self, id):
        idx = self.ids.index(id)
        return self.syntax[idx]

    def add_check_outputs(self, check_output):
        self.syntax.append(check_output.syntax)
        self.num_correct_count[check_output.num_correct] = \
            self.num_correct_count.get(check_output.num_correct, 0) + 1
        self.demo_correctness.append(check_output.demo_correctness)
        self.programs.append(check_output.program)
        self.ids.append(check_output.data_id)

    def summary_results(self):
        self.syntax_accuracy = float(self.syntax.count(True)) / len(self.syntax)
        self.num_correct_histogram = np.zeros(
            [max(self.num_correct_count) + 1], dtype=np.float)
        for i in range(len(self.num_correct_histogram)):
            self.num_correct_histogram[i] = \
                self.num_correct_count.get(i, 0)
        self.num_correct_histogram /= \
            np.array(self.num_correct_count.values()).astype(np.float).sum()

    def result_string(self):
        histogram_str = \
            ', '.join(['{:.3f}'.format(h) for h in self.num_correct_histogram])
        string = """
        **{name}**
        syntax_accuracy: {syntax_accuracy: .3f}
        num_correct_histogram: [{histogram}]
        """.format(name=self.name,
                   syntax_accuracy=self.syntax_accuracy,
                   histogram=histogram_str)
        return string


if __name__ == '__main__':
    args = GetArgument()

    if not os.path.exists(args.data_hdf5):
        raise ValueError(
            "data_hdf5 doesn't exist: {}".format(args.data_hdf5))

    if not os.path.exists(args.output_hdf5):
        raise ValueError(
            "output_path doesn't exist: {}".format(args.output_hdf5))

    with h5py.File(args.data_hdf5, 'r') as file_data:
        with h5py.File(args.output_hdf5, 'r') as file_output:
            data_info = file_data['data_info']
            num_train_demo = data_info['num_demo_per_program'].value
            num_test_demo = data_info['num_test_demo_per_program'].value
            dsl_type = data_info['dsl_type'].value
            dsl = get_KarelDSL(dsl_type=dsl_type, seed=123)
            karel_world = karel.Karel_world()

            # tf means "Teacher Forcing" and greedy means "Greedy Unrolling"
            results = {
                'train_tf_result': EvaluationResult('train_tf_result'),
                'test_tf_result': EvaluationResult('test_tf_result'),
                'train_greedy_result': EvaluationResult('train_greedy_result'),
                'test_greedy_result': EvaluationResult('test_greedy_result')
            }

            tf_syntax = []
            greedy_syntax = []
            for data_id in tqdm(file_output.keys()):
                data = file_data[data_id]
                output = file_output[data_id]
                gt_program = dsl.intseq2str(data['program'].value)
                tf_program = output['program_prediction'].value
                greedy_program = output['greedy_prediction'].value

                # Train demo
                train_tf_out = CheckProgram(
                    tf_program, data_id, num_train_demo,
                    data['s_h'].value, data['s_h_len'].value,
                    dsl, karel_world)
                results['train_tf_result'].add_check_outputs(train_tf_out)

                train_greedy_out = CheckProgram(
                    greedy_program, data_id, num_train_demo,
                    data['s_h'], data['s_h_len'],
                    dsl, karel_world)
                results['train_greedy_result'].add_check_outputs(train_greedy_out)

                # Test demo
                test_tf_out = CheckProgram(
                    tf_program, data_id, num_test_demo,
                    data['test_s_h'], data['test_s_h_len'],
                    dsl, karel_world)
                results['test_tf_result'].add_check_outputs(test_tf_out)

                test_greedy_out = CheckProgram(
                    greedy_program, data_id, num_test_demo,
                    data['test_s_h'], data['test_s_h_len'],
                    dsl, karel_world)
                results['test_greedy_result'].add_check_outputs(test_greedy_out)

            for result in results.values():
                result.summary_results()
                print(result.result_string())

            if args.log:
                if args.output_log_path is None:
                    args.output_log_path = "{}.eval_exe.log".format(
                        args.output_hdf5)
                with open(args.output_log_path, 'w') as f:
                    for result in results.values():
                        result.summary_results()
                        f.write(result.result_string())
            if args.dump:
                if args.new_hdf5_path is None:
                    args.new_hdf5_path = "{}.eval_exe.hdf5".format(
                        args.output_hdf5)
                with h5py.File(args.new_hdf5_path, 'w') as new_file:
                    print('Dump result files: {}'.format(args.new_hdf5_path))
                    tf_train = results['train_tf_result']
                    tf_test = results['test_tf_result']
                    greedy_train = results['train_greedy_result']
                    greedy_test = results['test_greedy_result']
                    for data_id in tqdm(file_output.keys()):
                        grp = new_file.create_group(data_id)
                        grp['program_prediction'] = \
                            tf_train.get_program_by_id(data_id)
                        grp['program_syntax'] = \
                            ('correct' if tf_train.get_syntax_by_id(data_id)
                             else 'wrong')
                        grp['program_is_correct_execution'] = \
                            tf_train.get_demo_correctness_by_id(data_id)
                        grp['program_num_execution_correct'] = \
                            (tf_train.get_demo_correctness_by_id(data_id)).astype(np.int32).sum()
                        grp['test_program_prediction'] = \
                            tf_test.get_program_by_id(data_id)
                        grp['test_program_syntax'] = \
                            ('correct' if tf_test.get_syntax_by_id(data_id)
                             else 'wrong')
                        grp['test_program_is_correct_execution'] = \
                            tf_test.get_demo_correctness_by_id(data_id)
                        grp['test_program_num_execution_correct'] = \
                            (tf_test.get_demo_correctness_by_id(data_id)).astype(np.int32).sum()
                        grp['greedy_prediction'] = \
                            greedy_train.get_program_by_id(data_id)
                        grp['greedy_syntax'] = \
                            ('correct' if greedy_train.get_syntax_by_id(data_id)
                             else 'wrong')
                        grp['greedy_is_correct_execution'] = \
                            greedy_train.get_demo_correctness_by_id(data_id)
                        grp['greedy_num_execution_correct'] = \
                            (greedy_train.get_demo_correctness_by_id(data_id)).astype(np.int32).sum()
                        grp['test_greedy_prediction'] = \
                            greedy_test.get_program_by_id(data_id)
                        grp['test_greedy_syntax'] = \
                            ('correct' if greedy_test.get_syntax_by_id(data_id)
                             else 'wrong')
                        grp['test_greedy_is_correct_execution'] = \
                            greedy_test.get_demo_correctness_by_id(data_id)
                        grp['test_greedy_num_execution_correct'] = \
                            (greedy_test.get_demo_correctness_by_id(data_id)).astype(np.int32).sum()
