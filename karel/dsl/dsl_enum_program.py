

def check_and_apply(queue, rule):
    r = rule[0].split()
    l = len(r)
    if len(queue) >= l:
        t = queue[-l:]
        if list(zip(*t)[0]) == r:
            new_t = rule[1](list(zip(*t)[1]))
            del queue[-l:]
            queue.extend(new_t)
            return True
    return False

rules = []

# k, n, s = fn(k, n)
# k: karel_world
# n: num_call
# s: success
# c: condition [True, False]
MAX_WHILE = 100


def r_prog(t):
    stmt = t[3]

    return [('prog', stmt(0, 0))]
rules.append(('DEF run m( stmt m)', r_prog))


def r_stmt(t):
    stmt = t[0]

    def fn(k, n):
        return stmt(k, n)
    return [('stmt', fn)]
rules.append(('while_stmt', r_stmt))
rules.append(('repeat_stmt', r_stmt))
rules.append(('stmt_stmt', r_stmt))
rules.append(('action', r_stmt))
rules.append(('if_stmt', r_stmt))
rules.append(('ifelse_stmt', r_stmt))


def r_stmt_stmt(t):
    stmt1, stmt2 = t[0], t[1]

    def fn(k, n):
        return stmt1(k, n) + stmt2(k, n)
    return [('stmt_stmt', fn)]
rules.append(('stmt stmt', r_stmt_stmt))


def r_if(t):
    cond, stmt = t[2], t[5]

    def fn(k, n):
        return ['if'] + cond(k, n) + stmt(k, n)
    return [('if_stmt', fn)]
rules.append(('IF c( cond c) i( stmt i)', r_if))


def r_ifelse(t):
    cond, stmt1, stmt2 = t[2], t[5], t[9]

    def fn(k, n):
        stmt1_out = stmt1(k, n)
        stmt2_out = stmt2(k, n)
        if stmt1_out == stmt2_out:
            return stmt1_out
        cond_out = cond(k, n)
        if cond_out[0] == 'not':
            else_cond = ['if'] + cond_out[1:]
        else:
            else_cond = ['if', 'not'] + cond_out
        return ['if'] + cond_out + stmt1_out + else_cond + stmt2_out
    return [('ifelse_stmt', fn)]
rules.append(('IFELSE c( cond c) i( stmt i) ELSE e( stmt e)', r_ifelse))


def r_while(t):
    cond, stmt = t[2], t[5]

    def fn(k, n):
        cond_out = cond(k, n)
        stmt_out = stmt(k, n)
        while_out = []
        for _ in range(MAX_WHILE):
            while_out.extend(['if'] + cond_out + stmt_out)
        return while_out
    return [('while_stmt', fn)]
rules.append(('WHILE c( cond c) w( stmt w)', r_while))


def r_repeat(t):
    cste, stmt = t[1], t[3]

    def fn(k, n):
        repeat_out = []
        for _ in range(cste()):
            repeat_out.extend(stmt(k, n))
        return repeat_out
    return [('repeat_stmt', fn)]
rules.append(('REPEAT cste r( stmt r)', r_repeat))


def r_cond1(t):
    cond = t[0]

    def fn(k, n):
        return cond(k, n)
    return [('cond', fn)]
rules.append(('cond_without_not', r_cond1))


def r_cond2(t):
    cond = t[2]

    def fn(k, n):
        cond_out = cond(k, n)
        if cond_out[0] == 'not':
            cond_out = cond_out[1:]
        else:
            cond_out = ['not'] + cond_out
        return cond_out
    return [('cond', fn)]
rules.append(('not c( cond c)', r_cond2))


def r_cond_without_not1(t):
    def fn(k, n):
        return ['frontIsClear']
    return [('cond_without_not', fn)]
rules.append(('frontIsClear', r_cond_without_not1))


def r_cond_without_not2(t):
    def fn(k, n):
        return ['leftIsClear']
    return [('cond_without_not', fn)]
rules.append(('leftIsClear', r_cond_without_not2))


def r_cond_without_not3(t):
    def fn(k, n):
        return ['rightIsClear']
    return [('cond_without_not', fn)]
rules.append(('rightIsClear', r_cond_without_not3))


def r_cond_without_not4(t):
    def fn(k, n):
        return ['markersPresent']
    return [('cond_without_not', fn)]
rules.append(('markersPresent', r_cond_without_not4))


def r_cond_without_not5(t):
    def fn(k, n):
        return ['not', 'markersPresent']
    return [('cond_without_not', fn)]
rules.append(('noMarkersPresent', r_cond_without_not5))


def r_action1(t):
    def fn(k, n):
        return ['move']
    return [('action', fn)]
rules.append(('move', r_action1))


def r_action2(t):
    def fn(k, n):
        return ['turnLeft']
    return [('action', fn)]
rules.append(('turnLeft', r_action2))


def r_action3(t):
    def fn(k, n):
        return ['turnRight']
    return [('action', fn)]
rules.append(('turnRight', r_action3))


def r_action4(t):
    def fn(k, n):
        return ['pickMarker']
    return [('action', fn)]
rules.append(('pickMarker', r_action4))


def r_action5(t):
    def fn(k, n):
        return ['putMarker']
    return [('action', fn)]
rules.append(('putMarker', r_action5))


def create_r_cste(number):
    def r_cste(t):
        return [('cste', lambda: number)]
    return r_cste
for i in range(20):
    rules.append(('R={}'.format(i), create_r_cste(i)))


def parse(program):
    p_tokens = program.split()[::-1]
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied: applied = False
        else:
            queue.append((p_tokens.pop(), None))
        for rule in rules:
            applied = check_and_apply(queue, rule)
            if applied: break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None, False
    return queue[0][1], True
