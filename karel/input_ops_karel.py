import numpy as np
import tensorflow as tf

from util import log


def check_data_id(dataset, data_id):
    if not data_id:
        return

    wrong = []
    for id in data_id:
        if id in dataset.data:
            pass
        else:
            wrong.append(id)

    if len(wrong) > 0:
        raise RuntimeError("There are %d invalid ids, including %s" % (
            len(wrong), wrong[:5]
        ))


def create_input_ops(dataset,
                     batch_size,
                     num_threads=16,           # for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True,
                     ):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}

    if data_id is None:
        data_id = dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device("/cpu:0"), tf.name_scope(scope):
        input_ops['id'] = tf.train.string_input_producer(
           tf.convert_to_tensor(data_id), capacity=128
        ).dequeue(name='input_ids_dequeue')

        p, pt, s, ts, a, at, ta, tat, pl, dl, tdl, per, tper \
            = dataset.get_data(data_id[0])

        def load_fn(id):
            # program [n, max_program_len]
            # program_tokens [max_program_len]
            # s_h [k, max_demo_len, h, w, 16]
            # test_s_h [test_k, max_demo_len, h, w, 16]
            # a_h [k, max_demo_len - 1, ac]
            # a_h_tokens [k, max_demo_len - 1]
            # test_a_h [test_k, max_demo_len - 1, ac]
            # test_a_h_tokens [test_k, max_demo_len - 1]
            # program_len [1]
            # demo_len [k]
            # test_demo_len [k]
            # per [k, t, c]
            # test_per [test_k, t, c]
            program, program_tokens, s_h, test_s_h, a_h, a_h_tokens, \
                test_a_h, test_a_h_tokens, program_len, demo_len, test_demo_len, \
                per, test_per = dataset.get_data(id)
            return (id, program.astype(np.float32), program_tokens.astype(np.int32),
                    s_h.astype(np.float32), test_s_h.astype(np.float32),
                    a_h.astype(np.float32), a_h_tokens.astype(np.int32),
                    test_a_h.astype(np.float32), test_a_h_tokens.astype(np.int32),
                    program_len.astype(np.float32), demo_len.astype(np.float32),
                    test_demo_len.astype(np.float32),
                    per.astype(np.float32), test_per.astype(np.float32))

        input_ops['id'], input_ops['program'], input_ops['program_tokens'], \
            input_ops['s_h'], input_ops['test_s_h'], \
            input_ops['a_h'], input_ops['a_h_tokens'], \
            input_ops['test_a_h'], input_ops['test_a_h_tokens'], \
            input_ops['program_len'], input_ops['demo_len'], \
            input_ops['test_demo_len'], input_ops['per'], input_ops['test_per'] = tf.py_func(
                load_fn, inp=[input_ops['id']],
                Tout=[tf.string, tf.float32, tf.int32, tf.float32, tf.float32,
                      tf.float32, tf.int32, tf.float32, tf.int32,
                      tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name='func_hp'
            )

        input_ops['id'].set_shape([])
        input_ops['program'].set_shape(list(p.shape))
        input_ops['program_tokens'].set_shape(list(pt.shape))
        input_ops['s_h'].set_shape(list(s.shape))
        input_ops['test_s_h'].set_shape(list(ts.shape))
        input_ops['a_h'].set_shape(list(a.shape))
        input_ops['a_h_tokens'].set_shape(list(at.shape))
        input_ops['test_a_h'].set_shape(list(ta.shape))
        input_ops['test_a_h_tokens'].set_shape(list(tat.shape))
        input_ops['program_len'].set_shape(list(pl.shape))
        input_ops['demo_len'].set_shape(list(dl.shape))
        input_ops['test_demo_len'].set_shape(list(tdl.shape))
        input_ops['per'].set_shape(list(per.shape))
        input_ops['test_per'].set_shape(list(tper.shape))

    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.75), 1024)

    if shuffle:
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_capacity,
        )
    else:
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

    return input_ops, batch_ops
