from dsl_parse import MONSTER_LIST, ITEMS_IN_INTEREST, ACTION_LIST, \
    DISTANCE_DICT, HORIZONTAL_DICT, CLEAR_DISTANCE_DICT, CLEAR_HORIZONTAL_DICT

SIMPLE_ACTION_LIST = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT',
                      'TURN_LEFT', 'TURN_RIGHT', 'ATTACK']
SIMPLE_PROGRAM_TOKENS = ['DEF', 'run', 'm(', 'm)', 'WHILE', 'c(', 'c)',
                         'w(', 'w)', 'IF', 'i(', 'i)', 'IFELSE', 'ELSE',
                         'e(', 'e)', 'not', 'EXIST', 'IN', 'INTARGET']

PROGRAM_TOKENS = ['DEF', 'run', 'm(', 'm)', 'WHILE', 'c(', 'c)', 'w(', 'w)',
                  'REPEAT', 'r(', 'r)', 'R=2', 'R=3', 'R=4', 'R=5', 'R=6',
                  'IF', 'i(', 'i)', 'IFELSE', 'ELSE', 'e(', 'e)', 'not', 'EXIST', 'IN', 'INTARGET',
                  'ISTHERE']


class VizDoomDSLVocab(object):
    def __init__(self, perception_type='clear', level='not_simple'):
        if perception_type == 'clear':
            distance_vocab = CLEAR_DISTANCE_DICT.keys()
            horizontal_vocab = CLEAR_HORIZONTAL_DICT.keys()
        elif perception_type == 'simple' or perception_type == 'more_simple':
            distance_vocab = []
            horizontal_vocab = []
        else:
            distance_vocab = DISTANCE_DICT.keys()
            horizontal_vocab = HORIZONTAL_DICT.keys()
        if level == 'simple':
            action_list = SIMPLE_ACTION_LIST
            program_tokens = SIMPLE_PROGRAM_TOKENS
        elif perception_type == 'simple':
            action_list = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT',
                           'MOVE_RIGHT', 'TURN_LEFT', 'TURN_RIGHT',
                           'ATTACK', 'SELECT_WEAPON1', 'SELECT_WEAPON3',
                           'SELECT_WEAPON4', 'SELECT_WEAPON5']
            program_tokens = ['DEF', 'run', 'm(', 'm)', 'WHILE', 'c(', 'c)',
                              'w(', 'w)', 'REPEAT', 'r(', 'r)', 'R=2', 'R=3',
                              'R=4', 'R=5', 'R=6', 'IF', 'i(', 'i)',
                              'IFELSE', 'ELSE', 'e(', 'e)', 'not',
                              'INTARGET', 'ISTHERE']
        elif perception_type == 'more_simple':
            action_list = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT',
                           'MOVE_RIGHT', 'TURN_LEFT', 'TURN_RIGHT',
                           'ATTACK', 'SELECT_WEAPON1', 'SELECT_WEAPON3',
                           'SELECT_WEAPON4', 'SELECT_WEAPON5']
            program_tokens = ['DEF', 'run', 'm(', 'm)', 'WHILE', 'c(', 'c)',
                              'w(', 'w)', 'REPEAT', 'r(', 'r)', 'R=2', 'R=3',
                              'R=4', 'R=5', 'R=6', 'IF', 'i(', 'i)',
                              'IFELSE', 'ELSE', 'e(', 'e)', 'not',
                              'ISTHERE']
        else:
            action_list = ACTION_LIST
            program_tokens = PROGRAM_TOKENS
        self.int2token = program_tokens + action_list + distance_vocab +\
            horizontal_vocab + MONSTER_LIST + ITEMS_IN_INTEREST
        self.token2int = {v: i for i, v in enumerate(self.int2token)}

        self.action_int2token = action_list
        self.action_token2int = {v: i for i, v in enumerate(self.action_int2token)}

    def str2intseq(self, string):
        return [self.token2int[t] for t in string.split()]

    def strlist2intseq(self, strlist):
        return [self.token2int[t] for t in strlist]

    def intseq2str(self, intseq):
        return ' '.join([self.int2token[i] for i in intseq])

    def token_dim(self):
        return len(self.int2token)

    def action_str2intseq(self, string):
        return [self.action_token2int[t] for t in string.split()]

    def action_intseq2str(self, intseq):
        return ' '.join([self.action_int2token[i] for i in intseq])

    def action_token_dim(self):
        return len(self.action_int2token)

    def action_strlist2intseq(self, strlist):
        return [self.action_token2int[t] for t in strlist]
