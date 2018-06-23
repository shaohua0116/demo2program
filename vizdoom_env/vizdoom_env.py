import math
import numpy as np

from vizdoom import DoomGame, ScreenResolution

from dsl.dsl_parse import MONSTER_LIST, ITEMS_IN_INTEREST, \
    DISTANCE_DICT, HORIZONTAL_DICT, CLEAR_DISTANCE_DICT, CLEAR_HORIZONTAL_DICT

PLAYER_NAME = ['DoomPlayer', 'MyPlayer']

DEAD_THINGS = ['GibbedMarine', 'DeadMarine', 'DeadZombieMan', 'DeadShotgunGuy',
               'DeadDoomImp', 'DeadDemon', 'DeadCacodemon', 'DeadLostSoul']

BLOODS = ['Blood', 'BloodSplatter', 'AxeBlood']

AMMO = ['Clip', 'ClipBox', 'RocketAmmo', 'RocketBox', 'Cell', 'CellPack', 'Shell',
        'ShellBox', 'BackPack']

CHEX_WEAPONS = ['Bootspoon', 'SuperBootspork', 'MiniZorcher', 'LargeZorcher',
                'SuperLargeZorcher', 'RapidZorcher', 'ZorchPropulsor',
                'PropulsorMissile', 'PhasingZorcher', 'PhaseZorchMissile',
                'LAZDevice', 'LAZBall']

DOOM_WEAPONS = ['Pistol', 'Chainsaw', 'Shotgun', 'SuperShotgun', 'Chaingun',
                'RocketLauncher', 'Rocket', 'Grenade', 'PlasmaRifle',
                'PlasmaBall', 'PlasmaBall1', 'PlasmaBall2', 'BFG9000',
                'BFGBall', 'BFGExtra']

DOOM_HEALTH = ['HealthBonus', 'Stimpack', 'Medikit']

DOOM_ARMOR = ['ArmorBonus', 'GreenArmor', 'BlueArmor']


# ACTION, TAKING, POST NONE, When to capture [1, 2]
FRAME_SKIP = {
    'NONE': [1, 1, 1],
    'MOVE_FORWARD': [5, 30, 1],
    'MOVE_BACKWARD': [5, 30, 1],
    'MOVE_LEFT': [5, 30, 1],
    'MOVE_RIGHT': [5, 30, 1],
    'TURN_LEFT': [5, 5, 1],
    'TURN_RIGHT': [5, 5, 1],
    'ATTACK': [1, 40, 0],
    'SELECT_WEAPON1': [40, 1, 1],
    'SELECT_WEAPON2': [40, 1, 1],
    'SELECT_WEAPON3': [40, 1, 1],
    'SELECT_WEAPON4': [40, 1, 1],
    'SELECT_WEAPON5': [40, 1, 1],
}

ATTACK_FRAME_SKIP = {
    1: 10,
    2: 5,
    3: 2,
    4: 2,
    5: 9,
}


class Vizdoom_env(object):
    def __init__(self, config='vizdoom_env/asset/default.cfg', verbose=False,
                 perception_type='more_simple'):
        self.verbose = verbose
        self.game = DoomGame()
        self.game.load_config(config)
        if self.verbose:
            self.game.set_window_visible(True)
            self.game.set_screen_resolution(ScreenResolution.RES_1280X960)

        self.game_variables = self.game.get_available_game_variables()
        self.buttons = self.game.get_available_buttons()
        self.action_strings = [b.__str__().replace('Button.', '')
                               for b in self.buttons]
        self.game_variable_strings = [v.__str__().replace('GameVariable.', '')
                                      for v in self.game_variables]
        self.perception_type = perception_type
        if perception_type == 'clear':
            self.distance_dict = CLEAR_DISTANCE_DICT
            self.horizontal_dict = CLEAR_HORIZONTAL_DICT
        elif perception_type == 'simple':
            pass
        elif perception_type == 'more_simple':
            pass
        else:
            self.distance_dict = DISTANCE_DICT
            self.horizontal_dict = HORIZONTAL_DICT

    def init_game(self):
        self.game.init()
        self.new_episode()

    def new_episode(self, init_state=None):
        self.game.new_episode()
        if init_state is not None:
            self.initialize_state(init_state)
        self.take_action('NONE')
        state = self.game.get_state()
        if state is None:
            raise RuntimeError('Cannot get initial states')
        img_arr = np.transpose(state.screen_buffer.copy(), [1, 2, 0])
        self.x_size = img_arr.shape[1]
        self.y_size = img_arr.shape[0]
        self.channel = img_arr.shape[2]
        self.get_state()
        if self.verbose:
            self.call_all_perception_primitives()
        p_v = self.get_perception_vector()
        self.s_h = [img_arr.copy()]
        self.a_h = []
        self.p_v_h = [p_v.copy()]  # perception vector

    def end_game(self):
        self.game.close()

    def state_transition(self, action_string):
        if action_string == 'NONE' or action_string in self.action_strings:
            self.take_action(action_string)
            self.a_h.append(action_string)
            if self.verbose:
                self.print_state()
            if FRAME_SKIP[action_string][2] == 0:
                self.get_state()
                self.s_h.append(self.screen.copy())
                p_v = self.get_perception_vector()
                self.p_v_h.append(p_v.copy())  # perception vector
            self.post_none(action_string)
            if FRAME_SKIP[action_string][2] == 1:
                self.get_state()
                self.s_h.append(self.screen.copy())
                p_v = self.get_perception_vector()
                self.p_v_h.append(p_v.copy())  # perception vector
            if self.verbose:
                self.call_all_perception_primitives()
        else:
            raise ValueError('Unknown action')

    def call_all_perception_primitives(self):
        for actor in MONSTER_LIST + ITEMS_IN_INTEREST:
            self.in_target(actor)
            for dist in self.distance_dict.keys():
                for horz in self.horizontal_dict.keys():
                    self.exist_actor_in_distance_horizontal(actor, dist, horz)
        for weapon_slot in range(1, 10):
            self.have_weapon(weapon_slot)
            self.have_ammo(weapon_slot)
            self.selected_weapon(weapon_slot)
        for actor in MONSTER_LIST:
            self.is_there(actor)
        self.no_selected_weapon_ammo()

    def take_action(self, action):
        action_vector = [a == action for a in self.action_strings]
        frame_skip = FRAME_SKIP[action][0]
        if action == 'ATTACK':
            state = self.game.get_state()
            gv_values = dict(zip(self.game_variable_strings,
                                 state.game_variables))
            weapon_num = int(gv_values['SELECTED_WEAPON'])
            frame_skip = ATTACK_FRAME_SKIP[weapon_num]
        self.game.make_action(action_vector, frame_skip)

    def post_none(self, action):
        none_vector = [a == 'NONE' for a in self.action_strings]
        self.game.make_action(none_vector, FRAME_SKIP[action][1])

    def get_action_list(self):
        return self.action_strings

    def init_actors(self):
        self.actors = {}

    def check_and_add_to_actors(self, actor_name, label):
        if actor_name not in self.actors:
            self.actors[actor_name] = []
        self.actors[actor_name].append(label)

    def get_actor_by_name(self, actor_name):
        if actor_name not in self.actors:
            self.actors[actor_name] = []
        return self.actors[actor_name]

    def get_state(self):
        state = self.game.get_state()
        if state is None:
            self.game_variables = dict()
            self.player = None
            self.monsters = []
            self.ammo = []
            self.init_actors()
            return
        self.game_variable_values = dict(zip(self.game_variable_strings, state.game_variables))
        self.monsters = []
        self.ammo = []
        self.weapons = []
        self.actors = {}
        for l in state.labels:
            if l.object_name in PLAYER_NAME:
                self.player = l
            elif l.object_name in MONSTER_LIST:
                self.monsters.append(l)
                self.check_and_add_to_actors(l.object_name, l)
            else:
                self.check_and_add_to_actors(l.object_name, l)

        self.labels = state.labels
        self.screen = np.transpose(state.screen_buffer, [1, 2, 0]).copy()

    def get_perception_vector_cond(self):
        if self.perception_type == 'simple' or \
                self.perception_type == 'more_simple':
            return self.get_perception_vector_cond_simple()
        else:
            return self.get_perception_vector_cond_basic()

    def get_perception_vector_cond_basic(self):
        vec = []
        for dist in self.distance_dict.keys():
            for horz in self.horizontal_dict.keys():
                for actor in MONSTER_LIST + ITEMS_IN_INTEREST:
                    vec.append('EXIST {} IN {} {}'.format(actor, dist, horz))
        for actor in MONSTER_LIST:
            vec.append('INTARGET {}'.format(actor))
        return vec

    def get_perception_vector_cond_simple(self):
        vec = []
        for actor in MONSTER_LIST:
            vec.append('ISTHERE {}'.format(actor))
        if self.perception_type == 'more_simple':
            return vec
        for actor in MONSTER_LIST:
            vec.append('INTARGET {}'.format(actor))
        return vec

    def get_perception_vector(self):
        if self.perception_type == 'simple' or\
                self.perception_type == 'more_simple':
            return self.get_perception_vector_simple()
        else: return self.get_perception_vector_basic()

    def get_perception_vector_basic(self):
        vec = []
        for dist in self.distance_dict.keys():
            for horz in self.horizontal_dict.keys():
                for actor in MONSTER_LIST + ITEMS_IN_INTEREST:
                    vec.append(self.exist_actor_in_distance_horizontal(actor, dist, horz))
        for actor in MONSTER_LIST:
            vec.append(self.in_target(actor))
        return np.array(vec)

    def get_perception_vector_simple(self):
        vec = []
        for actor in MONSTER_LIST:
            vec.append(self.is_there(actor))
        if self.perception_type == 'more_simple':
            return np.array(vec)
        for actor in MONSTER_LIST:
            vec.append(self.in_target(actor))
        return np.array(vec)

    def print_state(self):
        state = self.game.get_state()
        if state is None:
            print('No state')
            return
        game_variables = dict(zip(self.game_variable_strings, state.game_variables))
        game_variable_print = ''
        for key in sorted(game_variables.keys()):
            game_variable_print += '{}: {}, '.format(key, game_variables[key])
        game_variable_print += '\n'
        print(game_variable_print)
        for l in state.labels:
            print("id: {id}, name: {name}, position: [{pos_x},{pos_y},{pos_z}], "
                  "velocity: [{vel_x},{vel_y},{vel_z}], "
                  "angle: [{angle},{pitch},{roll}], "
                  "box: [{x},{y},{width},{height}]\n".format(
                      id=l.object_id, name=l.object_name,
                      pos_x=l.object_position_x, pos_y=l.object_position_y,
                      pos_z=l.object_position_z,
                      vel_x=l.object_velocity_x, vel_y=l.object_velocity_y,
                      vel_z=l.object_velocity_z,
                      angle=l.object_angle, pitch=l.object_pitch,
                      roll=l.object_roll,
                      x=l.x, y=l.y, width=l.width, height=l.height))

    def is_there(self, actor):
        if len(self.get_actor_by_name(actor)) > 0:
            if self.verbose: print('ISTHERE {}'.format(actor))
            return True
        else: return False

    def in_target(self, actor):
        center_x = self.x_size / 2
        center_y = self.y_size / 2
        for a in self.get_actor_by_name(actor):
            a_x_min, a_x_max = a.x, a.x + a.width
            a_y_min, a_y_max = a.y, a.y + a.height
            if center_x > a_x_min and center_x < a_x_max and\
                    center_y > a_y_min and center_y < a_y_max:
                        if self.verbose:
                            print('INTARGET {}'.format(actor))
                        return True
        return False

    def exist_actor_in_distance_horizontal(self, actor, dist, horz):
        cen_x = self.x_size / 2
        p = self.player
        for a in self.get_actor_by_name(actor):
            a_x_min, a_x_max = a.x, a.x + a.width
            d_x = a.object_position_x - p.object_position_x
            d_y = a.object_position_y - p.object_position_y
            d = math.sqrt(d_x**2 + d_y**2)
            if self.distance_dict[dist](d) and self.horizontal_dict[horz](a_x_min, a_x_max, cen_x):
                if self.verbose:
                    print('EXIST {} in {} {}'.format(actor, dist, horz))
                return True
        return False

    # Weapons
    # 1: Fist, chainsaw, 2: pistol, 3: shotgun, 4: chaingun, 5: rocket launcher, 6: plazma rifle
    # SELECT_WEAPON_1 switch between fist and chainsaw
    def have_weapon(self, weapon_slot):
        if self.game_variable_values['WEAPON{}'.format(weapon_slot)] > 0:
            if self.verbose:
                print('Have weapon {}'.format(weapon_slot))
            return True
        return False

    def have_ammo(self, weapon_slot):
        if weapon_slot == 1:  # Fist or Chainsaw
            if self.verbose:
                print('Have ammo {}'.format(weapon_slot))
            return True
        if self.game_variable_values['AMMO{}'.format(weapon_slot)] > 0:
            if self.verbose:
                print('Have ammo {}'.format(weapon_slot))
            return True
        return False

    def selected_weapon(self, weapon_slot):
        if self.game_variable_values['SELECTED_WEAPON'] == weapon_slot:
            if self.verbose:
                print('Weapon {} is selected'.format(weapon_slot))
            return True
        return False

    def no_selected_weapon_ammo(self):
        if self.game_variable_values['SELECTED_WEAPON_AMMO'] == 0:
            if self.verbose:
                print('no selected weapon ammo is left')
            return True
        return False

    def initialize_state(self, init_state):
        """ Takes random arguments and initialies the state

        Assumes that the max number of monster and ammo spawns is 5

        Params:
            init_state  [{"player_pos": [x, y], "monster_pos": [[x1, y1], [x2, y2]]}]
        """
        if 'player_pos' in init_state:
            x, y = init_state['player_pos']
            self.game.send_game_command('puke 20 {} {}'.format(x, y))
        if 'demon_pos' in init_state:
            for i, (x, y) in enumerate(init_state['demon_pos']):
                self.game.send_game_command(
                        'puke {} {} {}'.format(21 + i, x, y))
        if 'revenant_pos' in init_state:
            for i, (x, y) in enumerate(init_state['revenant_pos']):
                self.game.send_game_command(
                        'puke {} {} {}'.format(5 + i, x, y))
        if 'hellknight_pos' in init_state:
            for i, (x, y) in enumerate(init_state['hellknight_pos']):
                self.game.send_game_command(
                        'puke {} {} {}'.format(15 + i, x, y))
        if 'ammo_pos' in init_state:
            for i, (x, y) in enumerate(init_state['ammo_pos']):
                self.game.send_game_command(
                    'puke {} {} {}'.format(10 + i, x, y))
