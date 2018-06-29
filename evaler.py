from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import os
import time
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.util import log


class Evaler(object):

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'synthesis_baseline':
            from models.model_synthesis_baseline import Model
        elif model_name == 'induction_baseline':
            from models.model_induction_baseline import Model
        elif model_name == 'summarizer':
            from models.model_summarizer import Model
        elif model_name == 'full':
            from models.model_full import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.dataset_split = config.dataset_split
        self.train_dir = config.train_dir
        self.output_dir = getattr(config, 'output_dir',
                                  config.train_dir) or self.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        if config.dataset_type == 'karel':
            from karel_env.input_ops_karel import create_input_ops
        elif config.dataset_type == 'vizdoom':
            from vizdoom_env.input_ops_vizdoom import create_input_ops
        else:
            raise NotImplementedError("The dataset related code is not implemented.")

        self.dataset = dataset

        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov("Using Model class: %s", Model)
        self.model = Model(config, is_train=False)

        self.global_step = tf.contrib.framework.get_or_create_global_step(
            graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        # --- vars ---
        all_vars = tf.trainable_variables()
        log.warn("********* var ********** ")
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        tf.set_random_seed(123)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint = config.checkpoint
        if self.checkpoint is '' and self.train_dir:
            self.checkpoint = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint is '':
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            self.checkpoint_name = os.path.basename(self.checkpoint)
            log.info("Checkpoint path : %s", self.checkpoint)
        self.config.summary_file = self.checkpoint + '_report_testdata{}_num_k{}.txt'.format(
            self.config.max_steps * self.config.batch_size, self.config.num_k)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start Inference and Evaluation")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)
        try:
            if self.config.pred_program:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                log.infov("Output Dir: %s", self.output_dir)
                base_name = os.path.join(
                    self.output_dir,
                    'out_{}_{}'.format(self.checkpoint_name, self.dataset_split))
                text_file = open('{}.txt'.format(base_name), 'w')
                from karel_env.dsl import get_KarelDSL
                dsl = get_KarelDSL(dsl_type=self.dataset.dsl_type, seed=123)

                hdf5_file = h5py.File('{}.hdf5'.format(base_name), 'w')
                log_file = open('{}.log'.format(base_name), 'w')
            else:
                log_file = None

            if self.config.result_data:
                result_file = h5py.File(self.config.result_data_path, 'w')
                data_file = h5py.File(os.path.join(
                    self.config.dataset_path, 'data.hdf5'), 'r')

            if not self.config.no_loss:
                loss_all = []
                acc_all = []
                hist_all = {}
                time_all = []
                for s in xrange(self.config.max_steps):
                    step, loss, acc, hist, \
                        pred_program, pred_program_len, pred_is_correct_syntax, \
                        greedy_pred_program, greedy_program_len, greedy_is_correct_syntax, \
                        gt_program, gt_program_len, output, program_id, \
                        program_num_execution_correct, program_is_correct_execution, \
                        greedy_num_execution_correct, greedy_is_correct_execution, \
                        step_time = self.run_single_step(self.batch)
                    if not self.config.quiet:
                        step_msg = self.log_step_message(s, loss, acc,
                                                         hist, step_time)
                    if self.config.result_data:
                        for i in range(len(program_id)):
                            try:
                                grp = result_file.create_group(program_id[i])
                                grp['program'] = gt_program[i]
                                grp['pred_program'] = greedy_pred_program[i]
                                grp['pred_program_len'] = greedy_program_len[i][0]
                                grp['s_h'] = data_file[program_id[i]]['s_h'].value
                                grp['test_s_h'] = data_file[program_id[i]]['test_s_h'].value
                            except:
                                print('Duplicates: {}'.format(program_id[i]))
                                pass

                    # write pred/gt program
                    if self.config.pred_program:
                        log_file.write('{}\n'.format(step_msg))
                        for i in range(self.batch_size):
                            pred_program_token = np.argmax(
                                pred_program[i, :, :pred_program_len[i, 0]],
                                axis=0)
                            pred_program_str = dsl.intseq2str(pred_program_token)
                            greedy_program_token = np.argmax(
                                greedy_pred_program[i, :,
                                                    :greedy_program_len[i, 0]],
                                axis=0)
                            greedy_program_str = dsl.intseq2str(
                                greedy_program_token)
                            try: grp = hdf5_file.create_group(program_id[i])
                            except:
                                pass
                            else:
                                correctness = ['wrong', 'correct']
                                grp['program_prediction'] = pred_program_str
                                grp['program_syntax'] = \
                                    correctness[int(pred_is_correct_syntax[i])]
                                grp['program_num_execution_correct'] = \
                                    int(program_num_execution_correct[i])
                                grp['program_is_correct_execution'] = \
                                    program_is_correct_execution[i]
                                grp['greedy_prediction'] = \
                                    greedy_program_str
                                grp['greedy_syntax'] = \
                                    correctness[int(greedy_is_correct_syntax[i])]
                                grp['greedy_num_execution_correct'] = \
                                    int(greedy_num_execution_correct[i])
                                grp['greedy_is_correct_execution'] = \
                                    greedy_is_correct_execution[i]

                            text_file.write(
                                '[id: {}]\ngt: {}\npred{}: {}\ngreedy{}: {}\n'.format(
                                    program_id[i],
                                    dsl.intseq2str(np.argmax(
                                        gt_program[i, :, :gt_program_len[i, 0]], axis=0)),
                                    '(error)' if pred_is_correct_syntax[i] == 0 else '',
                                    pred_program_str,
                                    '(error)' if greedy_is_correct_syntax[i] == 0 else '',
                                    greedy_program_str,
                                ))
                    loss_all.append(np.array(loss.values()))
                    acc_all.append(np.array(acc.values()))
                    time_all.append(step_time)
                    for hist_key, hist_value in hist.items():
                        if hist_key not in hist_all:
                            hist_all[hist_key] = []
                        hist_all[hist_key].append(hist_value)

                loss_avg = np.average(np.stack(loss_all), axis=0)
                acc_avg = np.average(np.stack(acc_all), axis=0)
                hist_avg = {}
                for hist_key, hist_values in hist_all.items():
                    hist_avg[hist_key] = np.average(np.stack(hist_values), axis=0)
                final_msg = self.log_final_message(
                    loss_avg, loss.keys(), acc_avg,
                    acc.keys(), hist_avg, hist_avg.keys(), np.sum(time_all),
                    write_summary=self.config.write_summary,
                    summary_file=self.config.summary_file
                )

            if self.config.result_data:
                result_file.close()
                data_file.close()

            if self.config.pred_program:
                log_file.write('{}\n'.format(final_msg))
                log_file.write("Model class: {}\n".format(self.config.model))
                log_file.write("Checkpoint: {}\n".format(self.checkpoint))
                log_file.write("Dataset: {}\n".format(self.config.dataset_path))
                log_file.close()
                text_file.close()
                hdf5_file.close()

        except Exception as e:
            coord.request_stop(e)

        log.warning('Completed Evaluation.')

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, loss, acc, hist,
         pred_program, pred_program_len, pred_is_correct_syntax,
         greedy_pred_program, greedy_program_len, greedy_is_correct_syntax,
         gt_program, gt_program_len,
         program_num_execution_correct, program_is_correct_execution,
         greedy_num_execution_correct, greedy_is_correct_execution, output, _] = \
            self.session.run(
                [self.global_step, self.model.report_loss,
                 self.model.report_accuracy,
                 self.model.report_hist,
                 self.model.pred_program, self.model.program_len,
                 self.model.program_is_correct_syntax,
                 self.model.greedy_pred_program, self.model.greedy_pred_program_len,
                 self.model.greedy_program_is_correct_syntax,
                 self.model.ground_truth_program, self.model.program_len,
                 self.model.program_num_execution_correct,
                 self.model.program_is_correct_execution,
                 self.model.greedy_num_execution_correct,
                 self.model.greedy_is_correct_execution,
                 self.model.output,
                 self.step_op],
                feed_dict=self.model.get_feed_dict(batch_chunk)
            )

        _end_time = time.time()

        return step, loss, acc, hist, \
            pred_program, pred_program_len, pred_is_correct_syntax, \
            greedy_pred_program, greedy_program_len, greedy_is_correct_syntax, \
            gt_program, gt_program_len, output, batch_chunk['id'], \
            program_num_execution_correct, program_is_correct_execution, \
            greedy_num_execution_correct, greedy_is_correct_execution, \
            (_end_time - _start_time)

    def log_step_message(self, step, loss, acc, hist, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        loss_str = ""
        for k in sorted(loss.keys()):
            loss_str += "{}:{loss: .3f} ".format(k, loss=loss[k])
        acc_str = ""
        for k in sorted(acc.keys()):
            acc_str += "{}:{acc: .3f} ".format(k, acc=acc[k])
        hist_str = ""
        for k in sorted(hist.keys()):
            hist_str += "{}: [".format(k)
            for h in hist[k]:
                hist_str += "{acc: .3f}, ".format(acc=h)
            hist_str += "] "
        log_fn = (is_train and log.info or log.infov)
        msg = ("[{split_mode:5s} step {step:5d}] " +
               "{loss_str}" +
               "{acc_str}" +
               "{hist_str}" +
               "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} " +
               "instances/sec)"
               ).format(split_mode=(is_train and 'train' or 'val'),
                        step=step,
                        loss_str=loss_str,
                        acc_str=acc_str,
                        hist_str=hist_str,
                        sec_per_batch=step_time,
                        instance_per_sec=self.batch_size / step_time,
                        )
        log_fn(msg)
        return msg

    def log_final_message(self, loss, loss_key, acc, acc_key, hist, hist_key,
                          time, write_summary=False, summary_file=None, is_train=False):
        loss_str = ""
        for key, i in sorted(zip(loss_key, range(len(loss_key)))):
            loss_str += "{}:{loss: .3f} ".format(loss_key[i], loss=loss[i])
        acc_str = ""
        for key, i in sorted(zip(acc_key, range(len(acc_key)))):
            acc_str += "{}:{acc: .3f}\n".format(acc_key[i], acc=acc[i])
        hist_str = ""
        for key in sorted(hist_key):
            hist_str += "{}: [".format(key)
            for h in hist[key]:
                hist_str += "{acc: .3f}, ".format(acc=h)
            hist_str += "]\n"
        log_fn = (is_train and log.info or log.infov)
        msg = ("[Final Avg Report] \n" +
               "[Loss] {loss_str}\n" +
               "[Acc]  {acc_str}\n" +
               "[Hist] {hist_str}\n" +
               "[Time] ({time:.3f} sec)"
               ).format(split_mode=('Report'),
                        loss_str=loss_str,
                        acc_str=acc_str[:-1],
                        hist_str=hist_str[:-1],
                        time=time,
                        )
        log_fn(msg)
        log.infov("Model class: %s", self.config.model)
        log.infov("Checkpoint: %s", self.checkpoint)
        log.infov("Dataset: %s", self.config.dataset_path)
        if write_summary:
            final_msg = 'Model class: {}\nCheckpoint: {}\nDataset: %s {}\n{}'.format(
                self.config.model, self.checkpoint, self.config.dataset_path, msg)
            with open(summary_file, 'w') as f:
                f.write(final_msg)
        return msg


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='synthesis_baseline',
                        choices=['synthesis_baseline', 'induction_baseline',
                                 'summarizer', 'full'],
                        help='specify which type of models to evaluate')
    parser.add_argument('--dataset_type', type=str, default='karel',
                        choices=['karel', 'vizdoom'])
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/karel_dataset',
                        help='the path to your dataset')
    parser.add_argument('--dataset_split', type=str, default='test',
                        choices=['train', 'test', 'val'],
                        help='specify the data split to evaluate')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='the path to a trained checkpoint')
    parser.add_argument('--train_dir', type=str, default='',
                        help='the path to train_dir. '
                             'the newest checkpoint will be evaluated')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='the directory to write out programs')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='the number of batches to evaluate. '
                             'set to 0 to evaluate all testing data')
    # hyperparameters
    parser.add_argument('--num_k', type=int, default=10,
                        help='the number of seen demonstrations')
    parser.add_argument('--batch_size', type=int, default=20)
    # model hyperparameters
    parser.add_argument('--encoder_rnn_type', default='lstm',
                        choices=['lstm', 'rnn', 'gru'])
    parser.add_argument('--num_lstm_cell_units', type=int, default=512)
    parser.add_argument('--demo_aggregation', type=str, default='avgpool',
                        choices=['concat', 'avgpool', 'maxpool'],
                        help='how to aggregate the demo features')
    # evaluation task
    parser.add_argument('--no_loss', action='store_true', default=False,
                        help='set to True to not print out the accuracies and losses')
    parser.add_argument('--pred_program', action='store_true', default=False,
                        help='set to True to write out '
                             'predicted and ground truth programs')
    parser.add_argument('--result_data', action='store_true', default=False,
                        help='set to True to save evaluation results')
    parser.add_argument('--result_data_path', type=str, default='result.hdf5',
                        help='the file path to save evaluation results')
    # specify the ids of the testing data that you want to test
    parser.add_argument('--id_list', type=str,
                        help='specify the ids of the data points '
                             'that you want to evaluate. '
                             'By default a whole data split will be evaluated')
    # unseen test
    parser.add_argument('--unseen_test', action='store_true', default=False)
    # write summary file
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='set to True to not log out accuracies and losses '
                             'for every batch')
    parser.add_argument('--no_write_summary', action='store_true', default=False,
                        help='set to False to write out '
                             'the summary of accuracies and losses')
    parser.add_argument('--summary_file', type=str, default='report.txt',
                        help='the path to write the summary of accuracies and losses')
    config = parser.parse_args()

    config.write_summary = not config.no_write_summary

    if config.dataset_type == 'karel':
        import karel_env.dataset_karel as dataset
    elif config.dataset_type == 'vizdoom':
        import vizdoom_env.dataset_vizdoom as dataset
    else:
        raise ValueError(config.dataset)

    dataset_train, dataset_test, dataset_val = \
        dataset.create_default_splits(config.dataset_path,
                                        is_train=False, num_k=config.num_k)
    if config.dataset_split == 'train':
        target_dataset = dataset_train
    elif config.dataset_split == 'test':
        target_dataset = dataset_test
    elif config.dataset_split == 'val':
        target_dataset = dataset_val
    else:
        raise ValueError('Unknown dataset split')

    if not config.max_steps > 0:
        config.max_steps = int(len(target_dataset._ids)/config.batch_size)

    if config.dataset_type == 'karel':
        config.perception_type = ''
    elif config.dataset_type == 'vizdoom':
        config.perception_type = target_dataset.perception_type
    else:
        raise ValueError(config.dataset)
    # }}}

    # Data dim
    # [n, max_program_len], [max_program_len], [k, max_demo_len, h, w, depth]
    # [k, max_len_demo, ac], [1], [k]
    data_tuple = target_dataset.get_data(target_dataset.ids[0])
    program, _, s_h, test_s_h, a_h, _, _, _, program_len, demo_len, test_demo_len, \
        per, test_per = data_tuple[:13]

    config.dim_program_token = np.asarray(program.shape)[0]
    config.max_program_len = np.asarray(program.shape)[1]
    config.k = np.asarray(s_h.shape)[0]
    config.test_k = np.asarray(test_s_h.shape)[0]
    config.max_demo_len = np.asarray(s_h.shape)[1]
    config.h = np.asarray(s_h.shape)[2]
    config.w = np.asarray(s_h.shape)[3]
    config.depth = np.asarray(s_h.shape)[4]
    config.action_space = np.asarray(a_h.shape)[2]
    config.per_dim = np.asarray(per.shape)[2]
    if config.dataset_type == 'karel':
        config.dsl_type = target_dataset.dsl_type
        config.env_type = target_dataset.env_type
        config.vizdoom_pos_keys = []
        config.vizdoom_max_init_pos_len = -1
        config.level = None
    elif config.dataset_type == 'vizdoom':
        config.dsl_type = 'vizdoom_default'  # vizdoom has 1 dsl type for now
        config.env_type = 'vizdoom_default'  # vizdoom has 1 env type
        config.vizdoom_pos_keys = target_dataset.vizdoom_pos_keys
        config.vizdoom_max_init_pos_len = target_dataset.vizdoom_max_init_pos_len
        config.level = target_dataset.level

    evaler = Evaler(config, target_dataset)

    log.warning("dataset: %s", config.dataset_path)
    evaler.eval_run()

if __name__ == '__main__':
    main()
