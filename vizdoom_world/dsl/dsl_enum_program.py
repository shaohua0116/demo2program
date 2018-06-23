MONSTER_LIST = ['Demon', 'HellKnight', 'Revenant']

ITEMS_IN_INTEREST = ['MyAmmo']

ACTION_LIST = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT',
               'TURN_LEFT', 'TURN_RIGHT', 'ATTACK',
               'SELECT_WEAPON1', 'SELECT_WEAPON2', 'SELECT_WEAPON3',
               'SELECT_WEAPON4', 'SELECT_WEAPON5']

DISTANCE_DICT = {
    'doncare_dist': lambda d: True,
    'far': lambda d: d > 400,
    'mid': lambda d: d < 300,
    'close': lambda d: d < 180,
    'very_close': lambda d: d < 135}

HORIZONTAL_DICT = {
    'doncare_horz': lambda l, r, x: True,
    'center': lambda l, r, x: l < x and x < r,
    'slight_left': lambda l, r, x: r < x and x <= r + 10,
    'slight_right': lambda l, r, x: l > x and x >= l - 10,
    'mid_left': lambda l, r, x: r < x and x <= r + 20,
    'mid_right': lambda l, r, x: l > x and x >= l - 20,
    'left': lambda l, r, x: r < x,
    'right': lambda l, r, x: l > x}

CLEAR_DISTANCE_DICT = {
    'far': lambda d: d > 400,
    'mid_far': lambda d: 300 < d and d <= 400,
    'mid': lambda d: 180 < d and d <= 300,
    'close': lambda d: 135 < d and d <= 180,
    'very_close': lambda d: d <= 135}

CLEAR_HORIZONTAL_DICT = {
    'slight_left': lambda l, r, x: r < x and x <= r + 10,
    'slight_right': lambda l, r, x: l > x and x >= l - 10,
    'mid_left': lambda l, r, x: r + 10 < x and x <= r + 20,
    'mid_right': lambda l, r, x: l - 10 > x and x >= l - 20,
    'left': lambda l, r, x: r + 20 < x,
    'right': lambda l, r, x: l - 20 > x}

merge_distance_vocab = list(set(DISTANCE_DICT.keys()).union(
    set(CLEAR_DISTANCE_DICT.keys())))
merge_horizontal_vocab = list(set(HORIZONTAL_DICT.keys()).union(
    set(CLEAR_HORIZONTAL_DICT.keys())))


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

# world, n, s = fn(world, n)
# world: vizdoom_world
# n: num_call
# s: success
# c: condition [True, False]
MAX_WHILE = 1000


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
        return ['if'] + cond_out + stmt1(k, n) + else_cond + stmt2(k, n)
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
rules.append(('percept', r_cond1))


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


def r_percept1(t):
    actor, dist, horz = t[1], t[3], t[4]

    def fn(world, n):
        return ['exist_actor_in_distance_horizontal', actor(), dist(), horz()]
    return [('percept', fn)]
rules.append(('EXIST actor IN distance horizontal', r_percept1))


def r_percept2(t):
    actor = t[1]

    def fn(world, n):
        return ['in_target', actor()]
    return [('percept', fn)]
rules.append(('INTARGET actor', r_percept2))


def r_percept3(t):
    actor = t[1]

    def fn(world, n):
        return ['is_there', actor()]
    return [('percept', fn)]
rules.append(('ISTHERE actor', r_percept3))


def r_actor1(t):
    return [('actor', t[0])]
rules.append(('monster', r_actor1))


def create_r_monster(monster):
    def r_monster(t):
        return [('monster', lambda: monster)]
    return r_monster
for monster in MONSTER_LIST:
    rules.append((monster, create_r_monster(monster)))


def r_actor2(t):
    return [('actor', t[0])]
rules.append(('items', r_actor2))


def create_r_item(item):
    def r_item(t):
        return [('items', lambda: item)]
    return r_item
for item in ITEMS_IN_INTEREST:
    rules.append((item, create_r_item(item)))


def create_r_distance(distance):
    def r_distance(t):
        return [('distance', lambda: distance)]
    return r_distance
for distance in merge_distance_vocab:
    rules.append((distance, create_r_distance(distance)))


def create_r_horizontal(horizontal):
    def r_horizontal(t):
        return [('horizontal', lambda: horizontal)]
    return r_horizontal
for horizontal in merge_horizontal_vocab:
    rules.append((horizontal, create_r_horizontal(horizontal)))


def create_r_slot(slot_number):
    def r_slot(t):
        return [('slot', lambda: slot_number)]
    return r_slot
for slot_number in range(1, 7):
    rules.append(('S={}'.format(slot_number), create_r_slot(slot_number)))


def create_r_action(action):
    def r_action(t):
        def fn(world, n):
            return [action]
        return [('action', fn)]
    return r_action
for action in ACTION_LIST:
    rules.append((action, create_r_action(action)))


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
