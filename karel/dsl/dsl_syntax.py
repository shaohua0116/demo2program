from third_party import yacc
from dsl_base import KarelDSLBase


class KarelDSLSyntax(KarelDSLBase):
    def get_yacc(self):
        self.yacc, self.grammar = yacc.yacc(
            module=self,
            tabmodule="_parsetab_syntax",
            with_grammar=True)

    def get_next_candidates(self, code, **kwargs):
        next_candidates = self.yacc.parse(code, **kwargs)
        return next_candidates
