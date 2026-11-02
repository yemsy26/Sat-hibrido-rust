#!/usr/bin/env python3
"""Auditoria empirica de la particion en clases + ruptura de simetria.

Sin clausulas de Ramsey (s, t > n) la CNF de cada clase describe grafos
etiquetados. La union de los modelos de todas las clases debe contener
al menos un representante de CADA clase de isomorfismo de grafos de n
vertices (OEIS A000088), o de cada par {G, complemento} si s == t
((A000088 + A000171) / 2). Un control negativo (lex entre vecindarios
distintos, que no es una simetria) debe perder clases.
"""
import os
import sys

import graphs
import ramsey_cnf as rc
from common import ENUM, WORK, root_classes, run

GRAPHS = {4: 11, 5: 34, 6: 156, 7: 1044, 8: 12346}          # A000088
SELF_COMPL = {4: 1, 5: 2, 6: 0, 7: 0, 8: 10}                 # A000171


def models(path, n):
    out = run([ENUM, path, str(n * (n - 1) // 2)])
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def coverage(n, swap, lex, adjacent=False, card="totalizer", bad_pair=False):
    s, t = n + 1, (n + 1 if swap else n + 2)
    cl = graphs.Classifier(up_to_complement=swap)
    labeled = 0
    classes = [(None, None)] if lex == "full" else root_classes(s, t, n)
    for d, win in classes:
        cnf, nr = rc.build(s, t, n, root_degree=d, window=win, lex=lex,
                           adjacent=adjacent, card=card)
        if bad_pair and d is not None and 1 <= d < n - 1:
            cols = [k for k in range(n) if k not in (1, n - 1)]
            rc.lex_leq(cnf, [rc.edge_var(n, 1, k) for k in cols],
                       [rc.edge_var(n, n - 1, k) for k in cols])
        path = os.path.join(WORK, f"sym_{n}_{d}.cnf")
        rc.dump(path, s, t, n, cnf, nr)
        for bits in models(path, n):
            labeled += 1
            cl.add(graphs.from_bits(n, bits))
    expected = (GRAPHS[n] + SELF_COMPL[n]) // 2 if swap else GRAPHS[n]
    return cl.count, expected, labeled


def main():
    os.makedirs(WORK, exist_ok=True)
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    ok = True
    configs = [
        ("lex=root swap", dict(swap=True, lex="root")),
        ("lex=root sin swap", dict(swap=False, lex="root")),
        ("lex=root adyacente", dict(swap=True, lex="root", adjacent=True)),
        ("lex=root card=seq", dict(swap=True, lex="root", card="seq")),
        ("lex=full sin raiz", dict(swap=False, lex="full")),
        ("sin lex (referencia)", dict(swap=True, lex="none")),
    ]
    for n in range(4, nmax + 1):
        for name, kw in configs:
            if kw["lex"] == "none" and n > 6:
                continue  # demasiados modelos etiquetados; solo referencia
            got, exp, lab = coverage(n, **kw)
            flag = "OK" if got == exp else "FALLO"
            ok &= got == exp
            print(f"n={n} {name:22s} clases={got:6d} esperado={exp:6d} "
                  f"modelos_etiquetados={lab:7d} {flag}")
        got, exp, lab = coverage(n, swap=True, lex="root", bad_pair=True)
        flag = "OK (pierde clases)" if got < exp else "FALLO (no detecta)"
        ok &= got < exp
        print(f"n={n} {'control negativo':22s} clases={got:6d} esperado={exp:6d} "
              f"modelos_etiquetados={lab:7d} {flag}")
    print("TODO OK" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
