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
            new_t = rule[1](list(zip(*t)[1]), list(zip(*t)[2]))
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
MAX_FUNC_CALL = 100


def r_prog(tn, t):
    stmt = t[3]
    token_hit = tn[:3] + tn[4:]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False
        hit_s, n, s = stmt(world, n + 1)
        return token_hit + hit_s, n, s
    return [('prog', -1, fn)]
rules.append(('DEF run m( stmt m)', r_prog))


def r_stmt(tn, t):
    stmt = t[0]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return [], n, False
        return stmt(world, n + 1)
    return [('stmt', -1, fn)]
rules.append(('while_stmt', r_stmt))
rules.append(('repeat_stmt', r_stmt))
rules.append(('stmt_stmt', r_stmt))
rules.append(('action', r_stmt))
rules.append(('if_stmt', r_stmt))
rules.append(('ifelse_stmt', r_stmt))


def r_stmt_stmt(tn, t):
    stmt1, stmt2 = t[0], t[1]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return [], n, False
        hit_s1, n, s = stmt1(world, n + 1)
        if not s: return hit_s1, n, s
        if n > MAX_FUNC_CALL: return hit_s1, n, False
        hit_s2, n, s = stmt2(world, n)
        return hit_s1 + hit_s2, n, s
    return [('stmt_stmt', -1, fn)]
rules.append(('stmt stmt', r_stmt_stmt))


def r_if(tn, t):
    cond, stmt = t[2], t[5]
    token_hit = tn[:2] + tn[3:5] + tn[6:]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return [], n, False
        hit_c, n, s, c = cond(world, n + 1)
        if not s: return token_hit + hit_c, n, s
        if c:
            hit_s, n, s = stmt(world, n)
            return token_hit + hit_c + hit_s, n, s
        else: return token_hit + hit_c, n, s
    return [('if_stmt', -1, fn)]
rules.append(('IF c( cond c) i( stmt i)', r_if))


def r_ifelse(tn, t):
    cond, stmt1, stmt2 = t[2], t[5], t[9]
    token_hit = tn[:2] + tn[3:5] + tn[6:9] + tn[10:]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False
        hit_c, n, s, c = cond(world, n + 1)
        if not s: return token_hit + hit_c, n, s
        if c:
            hit_s1, n, s = stmt1(world, n)
            return token_hit + hit_c + hit_s1, n, s
        else:
            hit_s2, n, s = stmt2(world, n)
            return token_hit + hit_c + hit_s2, n, s
    return [('ifelse_stmt', -1, fn)]
rules.append(('IFELSE c( cond c) i( stmt i) ELSE e( stmt e)', r_ifelse))


def r_while(tn, t):
    cond, stmt = t[2], t[5]
    token_hit = tn[:2] + tn[3:5] + tn[6:]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False
        hit_c, n, s, c = cond(world, n)
        if not s: return token_hit + hit_c, n, s
        total_hit = token_hit
        while(c):
            hit_s, n, s = stmt(world, n)
            total_hit.extend(hit_s)
            if not s: return total_hit, n, s
            hit_c, n, s, c = cond(world, n)
            total_hit.extend(hit_c)
            if not s: return total_hit, n, s
        return total_hit, n, s
    return [('while_stmt', -1, fn)]
rules.append(('WHILE c( cond c) w( stmt w)', r_while))


def r_repeat(tn, t):
    cste, stmt = t[1], t[3]
    token_hit = tn[:3] + tn[4:]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False
        n += 1
        s = True
        total_hit = token_hit
        for _ in range(cste()):
            hit_s, n, s = stmt(world, n)
            total_hit.extend(hit_s)
            if not s: return total_hit, n, s
        return total_hit, n, s
    return [('repeat_stmt', -1, fn)]
rules.append(('REPEAT cste r( stmt r)', r_repeat))


def r_cond1(tn, t):
    cond = t[0]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return [], n, False, False
        return cond(world, n)
    return [('cond', -1, fn)]
rules.append(('percept', r_cond1))


def r_cond2(tn, t):
    cond = t[2]
    token_hit = tn[:2] + tn[3:]

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False, False
        hit_c, n, s, c = cond(world, n)
        return token_hit + hit_c, n, s, not c
    return [('cond', -1, fn)]
rules.append(('not c( cond c)', r_cond2))


def r_percept1(tn, t):
    actor, dist, horz = t[1], t[3], t[4]
    token_hit = tn

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False, False
        c = world.exist_actor_in_distance_horizontal(actor(), dist(), horz())
        return token_hit, n, True, c
    return [('percept', -1, fn)]
rules.append(('EXIST actor IN distance horizontal', r_percept1))


def r_percept2(tn, t):
    actor = t[1]
    token_hit = tn

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False, False
        c = world.in_target(actor())
        return token_hit, n, True, c
    return [('percept', -1, fn)]
rules.append(('INTARGET actor', r_percept2))


def r_percept3(tn, t):
    actor = t[1]
    token_hit = tn

    def fn(world, n):
        if n > MAX_FUNC_CALL: return token_hit, n, False, False
        c = world.is_there(actor())
        return token_hit,  n, True, c
    return [('percept', -1, fn)]
rules.append(('ISTHERE actor', r_percept3))


def r_actor1(tn, t):
    return [('actor', tn[0], t[0])]
rules.append(('monster', r_actor1))


def create_r_monster(monster):
    def r_monster(tn, t):
        return [('monster', tn[0], lambda: monster)]
    return r_monster
for monster in MONSTER_LIST:
    rules.append((monster, create_r_monster(monster)))


def r_actor2(tn, t):
    return [('actor', tn[0], t[0])]
rules.append(('items', r_actor2))


def create_r_item(item):
    def r_item(tn, t):
        return [('items', tn[0], lambda: item)]
    return r_item
for item in ITEMS_IN_INTEREST:
    rules.append((item, create_r_item(item)))


def create_r_distance(distance):
    def r_distance(tn, t):
        return [('distance', tn[0], lambda: distance)]
    return r_distance
for distance in merge_distance_vocab:
    rules.append((distance, create_r_distance(distance)))


def create_r_horizontal(horizontal):
    def r_horizontal(tn, t):
        return [('horizontal', tn[0], lambda: horizontal)]
    return r_horizontal
for horizontal in merge_horizontal_vocab:
    rules.append((horizontal, create_r_horizontal(horizontal)))


def create_r_slot(slot_number):
    def r_slot(tn, t):
        return [('slot', tn[0], lambda: slot_number)]
    return r_slot
for slot_number in range(1, 7):
    rules.append(('S={}'.format(slot_number), create_r_slot(slot_number)))


def create_r_action(action):
    def r_action(tn, t):
        token_hit = tn

        def fn(world, n):
            if n > MAX_FUNC_CALL: token_hit, n, False
            try: world.state_transition(action)
            except: return token_hit, n, False
            else: return token_hit, n, True
        return [('action', -1, fn)]
    return r_action
for action in ACTION_LIST:
    rules.append((action, create_r_action(action)))


def create_r_cste(number):
    def r_cste(tn, t):
        return [('cste', tn[0], lambda: number)]
    return r_cste
for i in range(20):
    rules.append(('R={}'.format(i), create_r_cste(i)))


def hit_count(program):
    p_tokens = program.split()[::-1]
    token_nums = list(range(len(p_tokens)))[::-1]
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied: applied = False
        else:
            queue.append((p_tokens.pop(), token_nums.pop(), None))
        for rule in rules:
            applied = check_and_apply(queue, rule)
            if applied: break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None, False
    return queue[0][2], True
