#!/usr/bin/env python3
"""Prueba exhaustiva de las codificaciones auxiliares de ramsey_cnf.

Para vectores pequenos se enumeran todos los modelos (proyectados en las
variables de entrada) y se comparan con la semantica esperada:
  degree_bounds: lo <= #verdaderos <= hi      (totalizador y secuencial)
  lex_leq:       a <=_lex b
"""
import itertools
import os
import sys

import ramsey_cnf as rc
from common import ENUM, WORK, run


def projected_models(cnf, nin, tag):
    path = os.path.join(WORK, f"enc_{tag}.cnf")
    with open(path, "w", newline="\n") as f:
        f.write(f"p cnf {cnf.nvars} {len(cnf.clauses)}\n")
        for cl in cnf.clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")
    out = run([ENUM, path, str(nin)])
    assert out.returncode == 0, out.stderr
    os.remove(path)
    return {tuple(c == "1" for c in line) for line in out.stdout.split()}


def fresh(nin):
    cnf = rc.Cnf(2)
    cnf.nvars = nin
    return cnf


def main():
    os.makedirs(WORK, exist_ok=True)
    fails = checks = 0
    for m in range(1, 8):
        lits = list(range(1, m + 1))
        for lo in range(0, m + 1):
            for hi in range(lo, m + 1):
                want = {x for x in itertools.product([False, True], repeat=m)
                        if lo <= sum(x) <= hi}
                for card in ("totalizer", "seq"):
                    cnf = fresh(m)
                    rc.degree_bounds(cnf, lits, lo, hi, card)
                    checks += 1
                    if projected_models(cnf, m, "card") != want:
                        fails += 1
                        print(f"FALLO card={card} m={m} lo={lo} hi={hi}")
    for L in range(1, 6):
        cnf = fresh(2 * L)
        rc.lex_leq(cnf, list(range(1, L + 1)), list(range(L + 1, 2 * L + 1)))
        want = {x for x in itertools.product([False, True], repeat=2 * L)
                if x[:L] <= x[L:]}
        checks += 1
        if projected_models(cnf, 2 * L, "lex") != want:
            fails += 1
            print(f"FALLO lex L={L}")
    print(f"{checks} comprobaciones, {fails} fallos")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
