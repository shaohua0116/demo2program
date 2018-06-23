import numpy as np
from karel import state_table


class color_code:
    HEADER = '\033[95m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    YELLOW = '\033[93m'
    CYAN = '\033[36m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def grid2str(grid):
    assert len(grid) == 16, 'Invalid representation of a grid'
    idx = np.argwhere(grid == np.amax(grid)).flatten().tolist()
    if len(idx) == 1:
        return state_table[idx[0]]
    elif len(idx) == 2:
        return '{} with {}'.format(state_table[idx[0]], state_table[idx[1]])
    else:
        return 'None'


# given a karel env state, return a symbol representation
def state2symbol(s):
    KAREL = "^>v<#"
    for i in range(s.shape[0]):
        str = ""
        for j in range(s.shape[1]):
            if np.sum(s[i, j, :4]) > 0 and np.sum(s[i, j, 6:]) > 0:
                idx = np.argmax(s[i, j])
                str += color_code.PURPLE+KAREL[idx]+color_code.END
            elif np.sum(s[i, j, :4]) > 0:
                idx = np.argmax(s[i, j])
                str += color_code.BLUE+KAREL[idx]+color_code.END
            elif np.sum(s[i, j, 4]) > 0:
                str += color_code.RED+'#'+color_code.END
            elif np.sum(s[i, j, 6:]) > 0:
                str += color_code.GREEN+'o'+color_code.END
            else:
                str += '.'
        print(str)
    return


# given a karel env state, return a visulized image
def state2image(s, grid_size=10, root_dir='./'):
    h = s.shape[0]
    w = s.shape[1]
    img = np.ones((h*grid_size, w*grid_size, 3))
    import h5py
    import os.path as osp
    f = h5py.File(osp.join(root_dir, 'asset/texture.hdf5'), 'r')
    wall_img = f['wall']
    marker_img = f['marker']
    # wall
    y, x = np.where(s[:, :, 4])
    for i in range(len(x)):
        img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img
    # marker
    y, x = np.where(np.sum(s[:, :, 6:], axis=-1))
    for i in range(len(x)):
        img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img
    # karel
    y, x = np.where(np.sum(s[:, :, :4], axis=-1))
    if len(y) == 1:
        y = y[0]
        x = x[0]
        idx = np.argmax(s[y, x])
        marker_present = np.sum(s[y, x, 6:]) > 0
        if marker_present:
            if idx == 0:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['n_m']
            elif idx == 1:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['e_m']
            elif idx == 2:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['s_m']
            elif idx == 3:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['w_m']
        else:
            if idx == 0:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['n']
            elif idx == 1:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['e']
            elif idx == 2:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['s']
            elif idx == 3:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['w']
    elif len(y) > 1:
        raise ValueError
    f.close()
    return img
