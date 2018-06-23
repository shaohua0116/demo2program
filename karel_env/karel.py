import numpy as np


MAX_NUM_MARKER = 10

state_table = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
    8: '3 markers',
    9: '4 markers',
    10: '5 markers',
    11: '6 markers',
    12: '7 markers',
    13: '8 markers',
    14: '9 markers',
    15: '10 markers'
}
action_table = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}


class Karel_world(object):

    def __init__(self, s=None, make_error=True):
        if s is not None:
            self.set_new_state(s)
        self.make_error = make_error

    def set_new_state(self, s):
        self.s = s.astype(np.bool)
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.h = self.s.shape[0]
        self.w = self.s.shape[1]
        p_v = self.get_perception_vector()
        self.p_v_h = [p_v.copy()]

    ###################################
    ###    Collect Demonstrations   ###
    ###################################
    def clear_history(self):
        self.s_h = [self.s.copy()]
        self.a_h = []

    def add_to_history(self, a_idx):
        self.s_h.append(self.s.copy())
        self.a_h.append(a_idx)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())

    # get location (x, y) and facing {north, east, south, west}
    def get_location(self):
        x, y, z = np.where(self.s[:, :, :4] > 0)
        return np.asarray([x[0], y[0], z[0]])

    # get the neighbor {front, left, right} loction
    def get_neighbor(self, face):
        loc = self.get_location()
        if face == 'front':
            neighbor_loc = loc[:2] + {
                0: [-1, 0],
                1: [0, 1],
                2: [1, 0],
                3: [0, -1]
            }[loc[2]]
        elif face == 'left':
            neighbor_loc = loc[:2] + {
                0: [0, -1],
                1: [-1, 0],
                2: [0, 1],
                3: [1, 0]
            }[loc[2]]
        elif face == 'right':
            neighbor_loc = loc[:2] + {
                0: [0, 1],
                1: [1, 0],
                2: [0, -1],
                3: [-1, 0]
            }[loc[2]]
        return neighbor_loc

    ###################################
    ###    Perception Primitives    ###
    ###################################
    # return if the neighbor {front, left, right} of Karel is clear
    def neighbor_is_clear(self, face):
        neighbor_loc = self.get_neighbor(face)
        if neighbor_loc[0] >= self.h or neighbor_loc[0] < 0 \
                or neighbor_loc[1] >= self.w or neighbor_loc[1] < 0:
            return False
        return not self.s[neighbor_loc[0], neighbor_loc[1], 4]

    def front_is_clear(self):
        return self.neighbor_is_clear('front')

    def left_is_clear(self):
        return self.neighbor_is_clear('left')

    def right_is_clear(self):
        return self.neighbor_is_clear('right')

    # return if there is a marker presented
    def marker_present(self):
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) > 0

    def no_marker_present(self):
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) == 0

    def get_perception_list(self):
        vec = ['frontIsClear', 'leftIsClear',
               'rightIsClear', 'markersPresent',
               'noMarkersPresent']
        return vec

    def get_perception_vector(self):
        vec = [self.front_is_clear(), self.left_is_clear(),
               self.right_is_clear(), self.marker_present(),
               self.no_marker_present()]
        return np.array(vec)

    ###################################
    ###       State Transition      ###
    ###################################
    # given a state and a action, return the next state
    def state_transition(self, a):
        a_idx = np.argmax(a)
        loc = self.get_location()

        if a_idx == 0:
            # move
            if self.front_is_clear():
                front_loc = self.get_neighbor('front')
                loc_vec = self.s[loc[0], loc[1], :4]
                self.s[front_loc[0], front_loc[1], :4] = loc_vec
                self.s[loc[0], loc[1], :4] = np.zeros(4) > 0
            else:
                if self.make_error:
                    raise RuntimeError("Failed to move.")
                loc_vec = np.zeros(4) > 0
                loc_vec[(loc[2] + 2) % 4] = True  # Turn 180
                self.s[loc[0], loc[1], :4] = loc_vec
            self.add_to_history(a_idx)
        elif a_idx == 1 or a_idx == 2:
            # turn left or right
            loc_vec = np.zeros(4) > 0
            loc_vec[(a_idx * 2 - 3 + loc[2]) % 4] = True
            self.s[loc[0], loc[1], :4] = loc_vec
            self.add_to_history(a_idx)

        elif a_idx == 3 or a_idx == 4:
            # pick up or put a marker
            num_marker = np.argmax(self.s[loc[0], loc[1], 5:])
            # just clip the num of markers for now
            # new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER-1)
            new_num_marker = a_idx*2-7 + num_marker
            if new_num_marker < 0:
                if self.make_error:
                    raise RuntimeError("No marker to pick up.")
                else:
                    new_num_marker = num_marker
            elif new_num_marker > MAX_NUM_MARKER-1:
                if self.make_error:
                    raise RuntimeError("Cannot put more marker.")
                else:
                    new_num_marker = num_marker
            marker_vec = np.zeros(MAX_NUM_MARKER+1) > 0
            marker_vec[new_num_marker] = True
            self.s[loc[0], loc[1], 5:] = marker_vec
            self.add_to_history(a_idx)
        else:
            raise RuntimeError("Invalid action")
        return
