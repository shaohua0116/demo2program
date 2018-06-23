from dsl_prob import KarelDSLProb

from dsl_prob_syntax import KarelDSLProbSyntax


def get_KarelDSL(dsl_type='prob', seed=None):
    if dsl_type == 'prob':
        return KarelDSLProb(seed=seed)
    else:
        raise ValueError('Undefined dsl type')


def get_KarelDSLSyntax(dsl_type='prob', seed=None):
    if dsl_type == 'prob':
        return KarelDSLProbSyntax(seed=seed)
    else:
        raise ValueError('Undefined dsl syntax type')
