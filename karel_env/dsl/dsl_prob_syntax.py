from dsl_syntax import KarelDSLSyntax


class KarelDSLProbSyntax(KarelDSLSyntax):
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

    INT_PREFIX = 'R='

    def t_INT(self, t):
        r'R=\d+'

        value = int(t.value.replace(self.INT_PREFIX, ''))
        if not (self.min_int <= value <= self.max_int):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`".
                            format(self.min_int, self.max_int, value))

        t.value = value
        return t

    def random_INT(self):
        return "{}{}".format(
            self.INT_PREFIX,
            self.rng.randint(self.min_int, self.max_int + 1))

    def t_error(self, t):
        t.lexer.skip(1)
        raise RuntimeError('Syntax Error')

    def p_prog(self, p):
        '''prog : prog_complete
                | prog_frag
        '''
        p[0] = [(str(i[0]).replace('\\', ''), i[1]) for i in p[1]]

    def p_prog_complete(self, p):
        '''prog_complete : prog4 M_RBRACE'''
        p[0] = []

    def p_prog_frag(self, p):
        '''prog_frag : prog1
                     | prog2
                     | prog3
                     | prog4
                     | prog4_frag
        '''
        p[0] = p[1]

    def p_prog1(self, p):
        '''prog1 : DEF'''
        p[0] = [(self.t_RUN, 4)]

    def p_prog2(self, p):
        '''prog2 : prog1 RUN'''
        p[0] = [(self.t_M_LBRACE, 3)]

    def p_prog3(self, p):
        '''prog3 : prog2 M_LBRACE'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_prog4(self, p):
        '''prog4 : prog3 stmt'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_stmt()] + \
            [(self.t_M_RBRACE, 1)]

    def next_token_stmt(self):
        next_token = \
            self.next_token_while() + \
            self.next_token_repeat() + \
            self.next_token_action() + \
            self.next_token_if() + \
            self.next_token_ifelse()
        return next_token

    def p_prog4_frag(self, p):
        '''prog4_frag : prog3 stmt_frag'''
        p[0] = [(i[0], i[1] + 1) for i in p[2]]

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        p[0] = self.next_token_stmt()

    def next_token_while(self):
        return [(self.t_WHILE, 7)]

    def next_token_repeat(self):
        return [(self.t_REPEAT, 5)]

    def next_token_action(self):
        return [(self.t_MOVE, 1), (self.t_TURN_RIGHT, 1), (self.t_TURN_LEFT, 1),
                (self.t_PICK_MARKER, 1), (self.t_PUT_MARKER, 1)]

    def next_token_if(self):
        return [(self.t_IF, 7)]

    def next_token_ifelse(self):
        return [(self.t_IFELSE, 11)]

    def p_while(self, p):
        '''while : while6 W_RBRACE
        '''
        p[0] = []

    def p_cond(self, p):
        '''cond : cond_without_not
                | cond_with_not
        '''
        p[0] = []

    def p_cond_with_not(self, p):
        '''cond_with_not : cond_with_not3 C_RBRACE
        '''
        p[0] = []

    def next_token_cond_without_not(self):
        return [(self.t_FRONT_IS_CLEAR, 1), (self.t_LEFT_IS_CLEAR, 1),
                (self.t_RIGHT_IS_CLEAR, 1), (self.t_MARKERS_PRESENT, 1),
                (self.t_NO_MARKERS_PRESENT, 1)]

    def next_token_cond_with_not(self):
        return [(self.t_NOT, 4)]

    def p_cond_without_not(self, p):
        '''cond_without_not : FRONT_IS_CLEAR
                            | LEFT_IS_CLEAR
                            | RIGHT_IS_CLEAR
                            | MARKERS_PRESENT
                            | NO_MARKERS_PRESENT
        '''
        p[0] = []

    def p_repeat(self, p):
        '''repeat : repeat4 R_RBRACE
        '''
        p[0] = []

    def p_cste(self, p):
        '''cste : INT
        '''
        p[0] = []

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        p[0] = []

    def p_action(self, p):
        '''action : MOVE
                  | TURN_RIGHT
                  | TURN_LEFT
                  | PICK_MARKER
                  | PUT_MARKER
        '''
        p[0] = []

    def p_if(self, p):
        '''if : if6 I_RBRACE
        '''
        p[0] = []

    def p_ifelse(self, p):
        '''ifelse : ifelse10 E_RBRACE
        '''
        p[0] = []

    def p_stmt_frag(self, p):
        '''stmt_frag : while_frag
                     | repeat_frag
                     | stmt_stmt_frag
                     | if_frag
                     | ifelse_frag
        '''
        p[0] = p[1]

    def p_while_frag(self, p):
        '''while_frag : while1
                      | while2
                      | while3
                      | while3_frag
                      | while4
                      | while5
                      | while6
                      | while6_frag
        '''
        p[0] = p[1]

    def p_while1(self, p):
        '''while1 : WHILE'''
        p[0] = [(self.t_C_LBRACE, 6)]

    def p_while2(self, p):
        '''while2 : while1 C_LBRACE'''
        p[0] = [(i[0], i[1] + 4) for i in self.next_token_cond()]

    def next_token_cond(self):
        return self.next_token_cond_without_not() + \
            self.next_token_cond_with_not()

    def p_while3(self, p):
        '''while3 : while2 cond'''
        p[0] = [(self.t_C_RBRACE, 4)]

    def p_while3_frag(self, p):
        '''while3_frag : while2 cond_frag'''
        p[0] = [(i[0], i[1] + 4) for i in p[2]]

    def p_while4(self, p):
        '''while4 : while3 C_RBRACE'''
        p[0] = [(self.t_W_LBRACE, 3)]

    def p_while5(self, p):
        '''while5 : while4 W_LBRACE'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_while6(self, p):
        '''while6 : while5 stmt'''
        p[0] = [(self.t_W_RBRACE, 1)] + \
            [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_while6_frag(self, p):
        '''while6_frag : while5 stmt_frag'''
        p[0] = [(i[0], i[1] + 1) for i in p[2]]

    def p_cond_frag(self, p):
        '''cond_frag : cond_with_not_frag
        '''
        p[0] = p[1]

    def p_cond_with_not_frag(self, p):
        '''cond_with_not_frag : cond_with_not1
                              | cond_with_not2
                              | cond_with_not3
        '''
        p[0] = p[1]

    def p_cond_with_not1(self, p):
        '''cond_with_not1 : NOT'''
        p[0] = [(self.t_C_LBRACE, 3)]

    def p_cond_with_not2(self, p):
        '''cond_with_not2 : cond_with_not1 C_LBRACE'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_cond_without_not()]

    def p_cond_with_not3(self, p):
        '''cond_with_not3 : cond_with_not2 cond_without_not'''
        p[0] = [(self.t_C_RBRACE, 1)]

    def p_repeat_frag(self, p):
        '''repeat_frag : repeat1
                       | repeat2
                       | repeat3
                       | repeat4
                       | repeat4_frag
        '''
        p[0] = p[1]

    def p_repeat1(self, p):
        '''repeat1 : REPEAT'''
        p[0] = [(i[0], i[1] + 3) for i in self.next_token_cste()]

    def next_token_cste(self):
        return [('R={}'.format(i), 1) for i
                in range(self.min_int, self.max_int + 1)]

    def p_repeat2(self, p):
        '''repeat2 : repeat1 cste'''
        p[0] = [(self.t_R_LBRACE, 3)]

    def p_repeat3(self, p):
        '''repeat3 : repeat2 R_LBRACE'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_repeat4(self, p):
        '''repeat4 : repeat3 stmt'''
        p[0] = [(self.t_R_RBRACE, 1)] + \
            [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_repeat4_frag(self, p):
        '''repeat4_frag : repeat3 stmt_frag'''
        p[0] = [(i[0], i[1] + 1) for i in p[2]]

    def p_stmt_stmt_frag(self, p):
        '''stmt_stmt_frag : stmt stmt_frag'''
        p[0] = p[2]

    def p_if_frag(self, p):
        '''if_frag : if1
                   | if2
                   | if3
                   | if3_frag
                   | if4
                   | if5
                   | if6
                   | if6_frag
        '''
        p[0] = p[1]

    def p_if1(self, p):
        '''if1 : IF'''
        p[0] = [(self.t_C_LBRACE, 6)]

    def p_if2(self, p):
        '''if2 : if1 C_LBRACE'''
        p[0] = [(i[0], i[1] + 4) for i in self.next_token_cond()]

    def p_if3(self, p):
        '''if3 : if2 cond'''
        p[0] = [(self.t_C_RBRACE, 4)]

    def p_if3_frag(self, p):
        '''if3_frag : if2 cond_frag'''
        p[0] = [(i[0], i[1] + 4) for i in p[2]]

    def p_if4(self, p):
        '''if4 : if3 C_RBRACE'''
        p[0] = [(self.t_I_LBRACE, 3)]

    def p_if5(self, p):
        '''if5 : if4 I_LBRACE'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_if6(self, p):
        '''if6 : if5 stmt'''
        p[0] = [(self.t_I_RBRACE, 1)] + \
            [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_if6_frag(self, p):
        '''if6_frag : if5 stmt_frag'''
        p[0] = [(i[0], i[1] + 1) for i in p[2]]

    def p_ifelse_frag(self, p):
        '''ifelse_frag : ifelse1
                       | ifelse2
                       | ifelse3
                       | ifelse3_frag
                       | ifelse4
                       | ifelse5
                       | ifelse6
                       | ifelse6_frag
                       | ifelse7
                       | ifelse8
                       | ifelse9
                       | ifelse10
                       | ifelse10_frag
        '''
        p[0] = p[1]

    def p_ifelse1(self, p):
        '''ifelse1 : IFELSE'''
        p[0] = [(self.t_C_LBRACE, 10)]

    def p_ifelse2(self, p):
        '''ifelse2 : ifelse1 C_LBRACE'''
        p[0] = [(i[0], i[1] + 8) for i in self.next_token_cond()]

    def p_ifelse3(self, p):
        '''ifelse3 : ifelse2 cond'''
        p[0] = [(self.t_C_RBRACE, 8)]

    def p_ifelse3_frag(self, p):
        '''ifelse3_frag : ifelse2 cond_frag'''
        p[0] = [(i[0], i[1] + 8) for i in p[2]]

    def p_ifelse4(self, p):
        '''ifelse4 : ifelse3 C_RBRACE'''
        p[0] = [(self.t_I_LBRACE, 7)]

    def p_ifelse5(self, p):
        '''ifelse5 : ifelse4 I_LBRACE'''
        p[0] = [(i[0], i[1] + 5) for i in self.next_token_stmt()]

    def p_ifelse6(self, p):
        '''ifelse6 : ifelse5 stmt'''
        p[0] = [(self.t_I_RBRACE, 5)] + \
            [(i[0], i[1] + 5) for i in self.next_token_stmt()]

    def p_ifelse6_frag(self, p):
        '''ifelse6_frag : ifelse5 stmt_frag'''
        p[0] = [(i[0], i[1] + 5) for i in p[2]]

    def p_ifelse7(self, p):
        '''ifelse7 : ifelse6 I_RBRACE'''
        p[0] = [(self.t_ELSE, 4)]

    def p_ifelse8(self, p):
        '''ifelse8 : ifelse7 ELSE'''
        p[0] = [(self.t_E_LBRACE, 3)]

    def p_ifelse9(self, p):
        '''ifelse9 : ifelse8 E_LBRACE'''
        p[0] = [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_ifelse10(self, p):
        '''ifelse10 : ifelse9 stmt'''
        p[0] = [(self.t_E_RBRACE, 1)] + \
            [(i[0], i[1] + 1) for i in self.next_token_stmt()]

    def p_ifelse10_frag(self, p):
        '''ifelse10_frag : ifelse9 stmt_frag'''
        p[0] = [(i[0], i[1] + 1) for i in p[2]]

    def p_error(self, p):
        raise RuntimeError('Syntax Error')
