#!/usr/bin/env python3
"""Generador CNF auditable para el problema de Ramsey (s, t, n).

Variable x(i,j) (i<j) verdadera  <=>  arista {i,j} de K_n es ROJA.
El fichero DIMACS tiene secciones en orden fijo, cada una documentada
en los comentarios de cabecera con su rango de clausulas:

  ramsey   C(n,s) clausulas (sin K_s rojo) + C(n,t) (sin K_t azul).
           Es la formula F; todo lo demas son restricciones de clase o
           ruptura de simetria.
  star     unitarias: el vertice 0 es rojo con 1..d y azul con d+1..n-1.
  degree   para todo vertice v != 0: lo <= grado_rojo(v) <= hi
           (totalizador con ambas direcciones, o contador secuencial).
  lex      Codish et al.: para pares i<j del mismo vecindario de la raiz
           (o de todo V si no hay raiz), fila_i \\ {i,j} <=_lex fila_j \\ {i,j}.

Las variables auxiliares (cardinalidad, lex) se numeran despues de las
C(n,2) variables de arista, que ocupan 1..C(n,2) en orden fila a fila.
"""
import argparse
import itertools
import math
import sys


def edge_var(n, i, j):
    """Numero DIMACS de la arista {i,j}; orden fila a fila (0,1),(0,2),..."""
    if i > j:
        i, j = j, i
    assert 0 <= i < j < n
    return i * n - i * (i + 1) // 2 + (j - i - 1) + 1


class Cnf:
    def __init__(self, n):
        self.n = n
        self.nvars = n * (n - 1) // 2
        self.clauses = []          # todo salvo la seccion ramsey
        self.sections = []         # (nombre, primera, ultima) en numeracion global
        self.notes = []

    def new_var(self):
        self.nvars += 1
        return self.nvars


# --------------------------------------------------------------- cardinalidad

def totalizer(cnf, lits, cap):
    """Totalizador (Bailleux-Boufkhad) con salidas truncadas a `cap`.

    Devuelve outs con outs[m-1] <-> (#lits verdaderos >= m), m=1..len(outs).
    Se emiten las dos direcciones, asi las salidas quedan *definidas*
    (extension definicional) y sirven tanto para <= como para >=.
    """
    if len(lits) == 1:
        return [lits[0]]
    mid = len(lits) // 2
    a = totalizer(cnf, lits[:mid], cap)
    b = totalizer(cnf, lits[mid:], cap)
    p, q = len(a), len(b)
    r = [cnf.new_var() for _ in range(min(p + q, cap))]
    R = len(r)
    for i in range(p + 1):
        for j in range(q + 1):
            if i + j >= 1:  # a>=i y b>=j  =>  r >= i+j
                cl = ([-a[i - 1]] if i else []) + ([-b[j - 1]] if j else [])
                cnf.clauses.append(cl + [r[min(i + j, R) - 1]])
            if i + j + 1 <= R:  # r >= i+j+1  =>  a>=i+1 o b>=j+1
                cl = ([a[i]] if i < p else []) + ([b[j]] if j < q else [])
                cnf.clauses.append(cl + [-r[i + j]])
    return r


def seq_at_most(cnf, lits, k):
    """Contador secuencial de Sinz (2005): sum(lits) <= k."""
    m = len(lits)
    if k >= m:
        return
    if k == 0:
        cnf.clauses.extend([[-x] for x in lits])
        return
    s = [[cnf.new_var() for _ in range(k)] for _ in range(m - 1)]
    cnf.clauses.append([-lits[0], s[0][0]])
    cnf.clauses.extend([[-s[0][j]] for j in range(1, k)])
    for i in range(1, m - 1):
        x = lits[i]
        cnf.clauses.append([-x, s[i][0]])
        cnf.clauses.append([-s[i - 1][0], s[i][0]])
        for j in range(1, k):
            cnf.clauses.append([-x, -s[i - 1][j - 1], s[i][j]])
            cnf.clauses.append([-s[i - 1][j], s[i][j]])
        cnf.clauses.append([-x, -s[i - 1][k - 1]])
    cnf.clauses.append([-lits[m - 1], -s[m - 2][k - 1]])


def degree_bounds(cnf, lits, lo, hi, card):
    m = len(lits)
    if lo > m or hi < 0 or lo > hi:
        cnf.clauses.append([])  # clase vacia: UNSAT trivial
        return
    if card == "totalizer":
        if lo <= 0 and hi >= m:
            return
        outs = totalizer(cnf, lits, min(hi + 1, m))
        if lo >= 1:
            cnf.clauses.append([outs[lo - 1]])
        if hi + 1 <= len(outs):
            cnf.clauses.append([-outs[hi]])
    else:
        seq_at_most(cnf, lits, hi)
        seq_at_most(cnf, [-x for x in lits], m - lo)   # >= lo


# ------------------------------------------------------------------------ lex

def lex_leq(cnf, a, b):
    """a <=_lex b (vectores de literales, 1=rojo > 0=azul).

    e_k = "prefijo 1..k igual" solo hace falta en sentido e_{k-1} y
    a_k=b_k => e_k; con a_k<=b_k impuesto, a_k=b_k equivale a
    (a_k o no b_k), lo que da las tres clausulas de 3 literales.
    """
    e_prev = None
    for k in range(len(a)):
        guard = [-e_prev] if e_prev is not None else []
        cnf.clauses.append(guard + [-a[k], b[k]])
        if k + 1 < len(a):
            e = cnf.new_var()
            cnf.clauses.append(guard + [-a[k], e])
            cnf.clauses.append(guard + [b[k], e])
            e_prev = e


def lex_pairs(n, root_degree, adjacent):
    if root_degree is None:
        blocks = [list(range(n))]
    else:
        blocks = [list(range(1, root_degree + 1)),
                  list(range(root_degree + 1, n))]
    for blk in blocks:
        if adjacent:
            yield from zip(blk, blk[1:])
        else:
            yield from itertools.combinations(blk, 2)


# ---------------------------------------------------------------------- build

def build(s, t, n, root_degree=None, window=None, lex="none",
          adjacent=False, card="totalizer"):
    if lex == "full" and root_degree is not None:
        raise ValueError("lex=full solo es correcto sin estrella de raiz")
    if lex == "root" and root_degree is None:
        raise ValueError("lex=root necesita --root-degree")
    cnf = Cnf(n)
    E = cnf.nvars
    n_ramsey = math.comb(n, s) + math.comb(n, t)
    cnf.sections.append(("ramsey", 1, n_ramsey))

    def mark(name, start):
        cnf.sections.append((name, n_ramsey + start + 1,
                             n_ramsey + len(cnf.clauses)))

    start = len(cnf.clauses)
    if root_degree is not None:
        for j in range(1, n):
            v = edge_var(n, 0, j)
            cnf.clauses.append([v] if j <= root_degree else [-v])
        mark("star", start)

    if window is not None:
        start = len(cnf.clauses)
        lo, hi = window
        first = 1 if root_degree is not None else 0
        for v in range(first, n):
            lits = [edge_var(n, v, u) for u in range(n) if u != v]
            degree_bounds(cnf, lits, lo, hi, card)
        mark("degree", start)

    if lex != "none":
        start = len(cnf.clauses)
        for i, j in lex_pairs(n, root_degree, adjacent):
            cols = [k for k in range(n) if k not in (i, j)]
            lex_leq(cnf, [edge_var(n, i, k) for k in cols],
                    [edge_var(n, j, k) for k in cols])
        mark("lex", start)

    cnf.notes = [
        f"ramsey s={s} t={t} n={n} edge_vars=1..{E}",
        f"root_degree={root_degree} window={window} lex={lex}"
        f" lex_pairs={'adjacent' if adjacent else 'all'} card={card}",
        "x(i,j)=edge_var fila a fila; verdadero=rojo",
    ]
    return cnf, n_ramsey


def ramsey_clauses(s, t, n):
    for c in itertools.combinations(range(n), s):
        yield [-edge_var(n, i, j) for i, j in itertools.combinations(c, 2)]
    for c in itertools.combinations(range(n), t):
        yield [edge_var(n, i, j) for i, j in itertools.combinations(c, 2)]


def write(path, s, t, n, **kw):
    cnf, n_ramsey = build(s, t, n, **kw)
    return dump(path, s, t, n, cnf, n_ramsey)


def dump(path, s, t, n, cnf, n_ramsey):
    total = n_ramsey + len(cnf.clauses)
    with open(path, "w", newline="\n") as f:
        for note in cnf.notes:
            f.write(f"c {note}\n")
        for name, a, b in cnf.sections:
            f.write(f"c section {name} clauses {a}..{b}\n")
        f.write(f"p cnf {cnf.nvars} {total}\n")
        buf = []
        for cl in itertools.chain(ramsey_clauses(s, t, n), cnf.clauses):
            buf.append(" ".join(map(str, cl)) + " 0\n")
            if len(buf) >= 65536:
                f.writelines(buf)
                buf.clear()
        f.writelines(buf)
    return cnf.nvars, total


# --------------------------------------------- comprobador semantico independiente

def coloring_from_model(n, model):
    """model: conjunto de literales verdaderos. Devuelve matriz booleana roja."""
    red = [[False] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        red[i][j] = red[j][i] = edge_var(n, i, j) in model
    return red


def mono_clique(red, k, color):
    """Busca un K_k monocromatico (color True=rojo) por backtracking simple."""
    n = len(red)

    def ext(clique, cands):
        if len(clique) == k:
            return clique
        for idx, v in enumerate(cands):
            nxt = [u for u in cands[idx + 1:] if red[v][u] == color]
            if len(clique) + 1 + len(nxt) >= k:
                r = ext(clique + [v], nxt)
                if r:
                    return r
        return None

    return ext([], list(range(n)))


def check_coloring(s, t, n, red, root_degree=None, window=None):
    """Verifica un modelo SIN usar la CNF: devuelve lista de fallos."""
    bad = []
    if (c := mono_clique(red, s, True)):
        bad.append(f"K{s} rojo {c}")
    if (c := mono_clique(red, t, False)):
        bad.append(f"K{t} azul {c}")
    deg = [sum(row) for row in red]
    if root_degree is not None:
        if [red[0][j] for j in range(1, n)] != [j <= root_degree for j in range(1, n)]:
            bad.append("estrella de raiz incorrecta")
    if window is not None:
        lo, hi = window
        if any(not lo <= d <= hi for d in deg):
            bad.append(f"grados fuera de ventana {deg}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("s", type=int)
    ap.add_argument("t", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--root-degree", type=int)
    ap.add_argument("--window", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--lex", choices=["none", "root", "full"], default="none")
    ap.add_argument("--lex-adjacent", action="store_true",
                    help="solo pares (i,i+1) en vez de todos los i<j")
    ap.add_argument("--card", choices=["totalizer", "seq"], default="totalizer")
    a = ap.parse_args()
    nv, nc = write(a.output, a.s, a.t, a.n, root_degree=a.root_degree,
                   window=tuple(a.window) if a.window else None, lex=a.lex,
                   adjacent=a.lex_adjacent, card=a.card)
    print(f"{a.output}: {nv} variables, {nc} clausulas", file=sys.stderr)


if __name__ == "__main__":
    main()
