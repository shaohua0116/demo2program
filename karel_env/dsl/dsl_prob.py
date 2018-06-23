"""
Domain specific language for Karel Environment

Code is adapted from https://github.com/carpedm20/karel
"""

import numpy as np

from dsl_base import KarelDSLBase, MIN_INT, MAX_INT, INT_PREFIX


class KarelDSLProb(KarelDSLBase):
    tokens = [
        'DEF', 'RUN', 'M_LBRACE', 'M_RBRACE',
        'MOVE', 'TURN_RIGHT', 'TURN_LEFT',
        'PICK_MARKER', 'PUT_MARKER',
        'R_LBRACE', 'R_RBRACE',
        'INT',  # 'NEWLINE', 'SEMI',
        'REPEAT',
        'C_LBRACE', 'C_RBRACE',
        'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
        'IF', 'IFELSE', 'ELSE',
        'FRONT_IS_CLEAR',  'LEFT_IS_CLEAR', 'RIGHT_IS_CLEAR',
        'MARKERS_PRESENT', 'NO_MARKERS_PRESENT',
        'NOT',
        'W_LBRACE', 'W_RBRACE',
        'WHILE',
    ]

    t_ignore = ' \t\n'

    t_M_LBRACE = 'm\('
    t_M_RBRACE = 'm\)'

    t_C_LBRACE = 'c\('
    t_C_RBRACE = 'c\)'

    t_R_LBRACE = 'r\('
    t_R_RBRACE = 'r\)'

    t_W_LBRACE = 'w\('
    t_W_RBRACE = 'w\)'

    t_I_LBRACE = 'i\('
    t_I_RBRACE = 'i\)'

    t_E_LBRACE = 'e\('
    t_E_RBRACE = 'e\)'

    t_DEF = 'DEF'
    t_RUN = 'run'
    t_WHILE = 'WHILE'
    t_REPEAT = 'REPEAT'
    t_IF = 'IF'
    t_IFELSE = 'IFELSE'
    t_ELSE = 'ELSE'
    t_NOT = 'not'

    t_FRONT_IS_CLEAR = 'frontIsClear'
    t_LEFT_IS_CLEAR = 'leftIsClear'
    t_RIGHT_IS_CLEAR = 'rightIsClear'
    t_MARKERS_PRESENT = 'markersPresent'
    t_NO_MARKERS_PRESENT = 'noMarkersPresent'

    conditional_functions = [
        t_FRONT_IS_CLEAR, t_LEFT_IS_CLEAR, t_RIGHT_IS_CLEAR,
        t_MARKERS_PRESENT, t_NO_MARKERS_PRESENT,
    ]

    t_MOVE = 'move'
    t_TURN_RIGHT = 'turnRight'
    t_TURN_LEFT = 'turnLeft'
    t_PICK_MARKER = 'pickMarker'
    t_PUT_MARKER = 'putMarker'

    action_functions = [
        t_MOVE,
        t_TURN_RIGHT, t_TURN_LEFT,
        t_PICK_MARKER, t_PUT_MARKER,
    ]

    #########
    # lexer
    #########

    def t_INT(self, t):
        r'R=\d+'

        value = int(t.value.replace(INT_PREFIX, ''))
        if not (MIN_INT <= value <= MAX_INT):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`".
                            format(MIN_INT, MAX_INT, value))

        t.value = value
        return t

    def random_INT(self):
        return "{}{}".format(
            INT_PREFIX,
            self.rng.randint(MIN_INT, MAX_INT + 1))

    def t_error(self, t):
        self.error = True
        t.lexer.skip(1)

    #########
    # parser
    #########

    prob_prog = [1.0]

    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]

        @self.callout
        def fn(karel_world):
            stmt(karel_world)
        p[0] = stmt

    prob_stmt = [0.1, 0.02, 0.7, 0.16, 0.01, 0.01]

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        function = p[1]

        @self.callout
        def fn(karel_world):
            function(karel_world)
        p[0] = fn

    prob_stmt_stmt = [1.0]

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]

        @self.callout
        def fn(karel_world):
            stmt1(karel_world)
            stmt2(karel_world)
        p[0] = fn

    prob_if = [1.0]

    def p_if(self, p):
        '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
        '''
        cond, stmt = p[3], p[6]

        @self.callout
        def fn(karel_world):
            condition = cond(karel_world)
            if condition != 'timeout' and condition:
                stmt(karel_world)

        p[0] = fn

    prob_ifelse = [1.0]

    def p_ifelse(self, p):
        '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]

        @self.callout
        def fn(karel_world):
            condition = cond(karel_world)
            if condition != 'timeout' and condition:
                stmt1(karel_world)
            elif condition != 'timeout':
                stmt2(karel_world)
            else:
                return

        p[0] = fn

    prob_while = [1.0]

    def p_while(self, p):
        '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE
        '''
        cond, stmt = p[3], p[6]

        @self.callout
        def fn(karel_world):
            condition = cond(karel_world)
            while(condition != 'timeout' and condition):
                stmt(karel_world)
                condition = cond(karel_world)

        p[0] = fn

    prob_repeat = [1.0]

    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt R_RBRACE
        '''
        cste, stmt = p[2], p[4]

        @self.callout
        def fn(karel_world):
            for _ in range(cste()):
                stmt(karel_world)

        p[0] = fn

    prob_cond = [0.9, 0.1]

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT C_LBRACE cond_without_not C_RBRACE
        '''
        if callable(p[1]):
            cond_without_not = p[1]

            def fn(karel_world):
                return cond_without_not(karel_world)
            p[0] = fn
        else:  # NOT
            cond_without_not = p[3]

            def fn(karel_world):
                return not cond_without_not(karel_world)
            p[0] = fn

    prob_cond_without_not = [0.7, 0.1, 0.1, 0.05, 0.05]

    def p_cond_without_not(self, p):
        '''cond_without_not : FRONT_IS_CLEAR
                            | LEFT_IS_CLEAR
                            | RIGHT_IS_CLEAR
                            | MARKERS_PRESENT
                            | NO_MARKERS_PRESENT
        '''
        cond_without_not = p[1]

        def fn(karel_world):
            if cond_without_not == self.t_FRONT_IS_CLEAR:
                return karel_world.front_is_clear()
            elif cond_without_not == self.t_LEFT_IS_CLEAR:
                return karel_world.left_is_clear()
            elif cond_without_not == self.t_RIGHT_IS_CLEAR:
                return karel_world.right_is_clear()
            elif cond_without_not == self.t_MARKERS_PRESENT:
                return karel_world.marker_present()
            elif cond_without_not == self.t_NO_MARKERS_PRESENT:
                return karel_world.no_marker_present()
            else:
                raise ValueError("No such condition")

        p[0] = fn

    prob_action = [0.7, 0.1, 0.1, 0.05, 0.05]

    def p_action(self, p):
        '''action : MOVE
                  | TURN_RIGHT
                  | TURN_LEFT
                  | PICK_MARKER
                  | PUT_MARKER
        '''
        action = p[1]

        def fn(karel_world):
            action_v = np.array([self.t_MOVE,
                                 self.t_TURN_LEFT, self.t_TURN_RIGHT,
                                 self.t_PICK_MARKER, self.t_PUT_MARKER]) == action
            karel_world.state_transition(action_v)
        p[0] = fn

    prob_cste = [1.0]

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        p[0] = lambda: int(value)

    def p_error(self, p):
        self.error = True

    def random_tokens(self, start_token="prog", depth=0, max_depth=6, nesting_depth=0, max_nesting_depth=4):
        if start_token == 'stmt':
            if nesting_depth > max_nesting_depth or depth > max_depth:
                start_token = "action"

        codes = []
        candidates = self.prodnames[start_token]
        sample_prob = getattr(self, 'prob_{}'.format(start_token))

        prod = candidates[self.rng.choice(range(len(candidates)), p=sample_prob)]

        for term in prod.prod:
            if term in self.prodnames:  # need digging
                if term in ['if', 'ifelse', 'repeat', 'while']:  # increase nested depth
                    codes.extend(self.random_tokens(term, depth + 1, max_depth, nesting_depth + 1, max_nesting_depth))
                else:
                    codes.extend(self.random_tokens(term, depth + 1, max_depth, nesting_depth, max_nesting_depth))
            else:
                token = getattr(self, 't_{}'.format(term))
                if callable(token):
                    if token == self.t_INT:
                        token = self.random_INT()
                    else:
                        raise Exception(" [!] Undefined token `{}`".format(token))

                codes.append(str(token).replace('\\', ''))

        return codes
