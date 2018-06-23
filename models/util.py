""" Utilities """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Logging
# =======

import logging
from colorlog import ColoredFormatter
import matplotlib.colors as cl
import numpy as np

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('attcap')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')


def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

logging.Logger.infov = _infov


def visualize_flow(x, y):
    img_batch = []
    h, w = x.shape[-2:]
    for i in range(x.shape[1]):
        img_time_step = []
        for j in range(x.shape[0]):
            du = x[j, i]
            dv = y[j, i]
            # valid = flow[:, :, 2]
            max_flow = max(np.max(du), np.max(dv))
            img = np.zeros((h, w, 3), dtype=np.float64)
            # angle layer
            img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
            # magnitude layer, normalized to 1
            img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
            # phase layer
            img[:, :, 2] = 8 - img[:, :, 1]
            # clip to [0,1]
            small_idx = img < 0
            large_idx = img > 1
            img[small_idx] = 0
            img[large_idx] = 1
            # convert to rgb
            img = cl.hsv_to_rgb(img)
            img_time_step.append(img)
        img_time_step = np.stack(img_time_step, axis=-1)
        img_time_step = np.transpose(img_time_step, [0, 1, 3, 2])
        img_time_step = np.reshape(img_time_step, [h, w*x.shape[0], 3])
        img_batch.append(img_time_step)
    return np.stack(img_batch, axis=0).astype(np.float32)
