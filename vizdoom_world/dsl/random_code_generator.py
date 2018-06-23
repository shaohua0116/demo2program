import numpy as np

from dsl_parse import parse

stmt_length_range = {
        'span0': (1, 3),  # [1, 6]
        'span1': (1, 2),  # [1, 2]
        'span2': (1, 2)}  # [1, 1]

rules = {}
rules['prog'] = []
rules['prog'].append(('DEF run m( stmt0 m)', 1))

rules['action'] = []
rules['action'].append(('MOVE_FORWARD', 0.1))
rules['action'].append(('MOVE_BACKWARD', 0.1))
rules['action'].append(('MOVE_LEFT', 0.2))
rules['action'].append(('MOVE_RIGHT', 0.2))
rules['action'].append(('TURN_LEFT', 0.1))
rules['action'].append(('TURN_RIGHT', 0.1))
rules['action'].append(('ATTACK', 0.2))

rules['stmt0'] = []
rules['stmt0'].append(('action_stmt1', 0.2))
rules['stmt0'].append(('if_stmt1', 0.25))
rules['stmt0'].append(('ifelse_stmt1', 0.24))
rules['stmt0'].append(('while_stmt1', 0.3))
rules['stmt0'].append(('repeat_stmt1', 0.01))

rules['stmt1'] = []
rules['stmt1'].append(('action_stmt1', 0.2))
rules['stmt1'].append(('if_stmt1', 0.25))
rules['stmt1'].append(('ifelse_stmt1', 0.24))
rules['stmt1'].append(('while_stmt1', 0.3))
rules['stmt1'].append(('repeat_stmt1', 0.01))

rules['stmt2'] = []
rules['stmt2'].append(('action', 0.8))
rules['stmt2'].append(('action action', 0.2))

rules['action_stmt1'] = []
rules['action_stmt1'].append(('action', 0.85))
rules['action_stmt1'].append(('action action', 0.1))
rules['action_stmt1'].append(('action action action', 0.05))

rules['if_stmt1'] = []
rules['if_stmt1'].append(('IF c( cond c) i( stmt2 i)', 1))

rules['ifelse_stmt1'] = []
rules['ifelse_stmt1'].append(
        ('IFELSE c( cond c) i( stmt2 i) ELSE e( stmt2 e)', 1))

rules['while_stmt1'] = []
rules['while_stmt1'].append(('WHILE c( cond c) w( stmt2 w)', 1))

rules['repeat_stmt1'] = []
rules['repeat_stmt1'].append(('REPEAT cste r( stmt2 r)', 1))

rules['if_stmt2'] = []
rules['if_stmt2'].append(('IF c( cond c) i( stmt1 i)', 1))

rules['ifelse_stmt2'] = []
rules['ifelse_stmt2'].append(
        ('IFELSE c( cond c) i( stmt1 i) ELSE e( stmt1 e)', 1))

rules['while_stmt2'] = []
rules['while_stmt2'].append(('WHILE c( cond c) w( stmt1 w)', 1))

rules['repeat_stmt2'] = []
rules['repeat_stmt2'].append(('REPEAT cste r( stmt1 r)', 1))

rules['cond'] = []
rules['cond'].append(('not c( percept c)', 0.2))
rules['cond'].append(('percept', 0.8))

rules['cste'] = []
rules['cste'].append(('R=2', 0.4))
rules['cste'].append(('R=3', 0.3))
rules['cste'].append(('R=4', 0.3))


# This generator handles single depth case only
class DoomProgramGenerator():
    def __init__(self, seed=123):
        self.rng = np.random.RandomState(seed)

    def get_percepts_value(self, world_list):
        percepts_value = []
        for world in world_list:
            percepts_value.append(world.get_perception_vector())
        percepts_value = np.stack(percepts_value).astype(np.float)
        return percepts_value

    def compute_percepts_prob(self, world_list):
        percepts_value = self.get_percepts_value(world_list)
        num_demo = float(len(world_list))
        percepts_sum = percepts_value.sum(axis=0)
        percepts_diff = (num_demo / 2.0 - abs(num_demo / 2.0 - percepts_sum))
        percepts_diff = percepts_diff ** 2
        if percepts_diff.sum() == 0:
            percepts_diff[:] += 1e-10
        percepts_prob = percepts_diff / percepts_diff.sum()
        return percepts_prob

    def random_expand_token(self, token, percepts, world_list, depth=0):
        # Expansion
        candidates, sample_prob = zip(*rules[token])
        sample_idx = self.rng.choice(range(len(candidates)), p=sample_prob)
        expansion = []
        for new_t in candidates[sample_idx].split():
            if new_t in ['stmt0', 'stmt1', 'stmt2']:
                stmt_len = self.rng.choice(
                    range(*stmt_length_range['span{}'.format(depth)]))
                expansion.extend([new_t] * stmt_len)
            else: expansion.append(new_t)
        codes = []
        for t in expansion:
            if t in rules:
                # Increase nested depth
                if t in ['stmt0', 'stmt1', 'stmt2']:
                    sub_codes, success = self.random_expand_token(t, percepts, world_list, depth + 1)
                    if not success:
                        return [], False
                    codes.extend(sub_codes)
                else:
                    sub_codes, success = self.random_expand_token(t, percepts, world_list, depth)
                    if not success:
                        return [], False
                    codes.extend(sub_codes)
            elif t == 'percept':
                percepts_prob = self.compute_percepts_prob(world_list)
                percept_idx = self.rng.choice(range(len(percepts)), p=percepts_prob)
                codes.append(percepts[percept_idx])
            else: codes.append(t)
        if token in ['action_stmt1', 'if_stmt1', 'ifelse_stmt1',
                     'while_stmt1', 'repeat_stmt1']:
            # run new statement to be capable of getting next statements
            stmt = ' '.join(codes)
            exe, compile_success = parse(stmt)
            if not compile_success:
                raise RuntimeError('Compile failure should not happen')
            for world in world_list:
                w, num_call, success = exe(world, 0)
                if not success:
                    return [], False

        return codes, True

    def random_code(self, percepts, world_list):
        codes, success = self.random_expand_token('prog', percepts, world_list, depth=0)
        return ' '.join(codes), success
