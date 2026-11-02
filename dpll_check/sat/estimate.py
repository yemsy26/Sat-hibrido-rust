#!/usr/bin/env python3
"""Estimacion del coste cube-and-conquer de una clase de raiz (Knuth).

Ejemplo (clase d=18 de R(5,5) en K43, cubos de 40 decisiones):
  python estimate.py 5 5 43 18 --window 18 24 --k 40 --samples 30 --leaf-time 60

Arbol de cubos: orden estatico de aristas (dentro del vecindario rojo de
la raiz, luego dentro del azul, luego cruzadas), poda por propagacion
unitaria; ver src/cube_estimate.cpp. Las hojas TIMEOUT cuentan como
leaf-time, por lo que el coste estimado es entonces una COTA INFERIOR.
"""
import argparse
import json
import math
import os
import statistics
import sys

import ramsey_cnf as rc
from common import HERE, RESULTS, WORK, run

ESTIMATOR = os.path.join(HERE, "bin", "cube_estimate" + (".exe" if os.name == "nt" else ""))


def edge_order(n, d):
    red, blue = range(1, d + 1), range(d + 1, n)
    inner = lambda blk: [rc.edge_var(n, i, j) for i in blk for j in blk if i < j]
    cross = [rc.edge_var(n, i, j) for i in red for j in blue]
    return inner(red) + inner(blue) + cross


def summarize(rows, leaf_time):
    w = [2.0 ** r["log2w"] for r in rows]
    cost = [wi * (leaf_time if r["res"] == "TIMEOUT" else r["secs"])
            for wi, r in zip(w, rows)]
    m = len(rows)

    def mean_se(xs):
        mu = sum(xs) / m
        se = statistics.stdev(xs) / math.sqrt(m) if m > 1 else float("nan")
        return mu, se

    leaves = [wi if r["res"] != "PRUNED" else 0.0 for wi, r in zip(w, rows)]
    res = {k: sum(r["res"] == k for r in rows)
           for k in ("UNSAT", "SAT", "TIMEOUT", "PRUNED")}
    lmu, lse = mean_se(leaves)
    cmu, cse = mean_se(cost)
    solved = [r["secs"] for r in rows if r["res"] in ("UNSAT", "SAT")]
    return {"samples": m, "outcomes": res, "leaves_est": lmu, "leaves_se": lse,
            "cost_s_est": cmu, "cost_s_se": cse,
            "cost_is_lower_bound": res["TIMEOUT"] > 0,
            "mean_leaf_s_solved": (sum(solved) / len(solved)) if solved else None,
            "log2w_mean": sum(r["log2w"] for r in rows) / m,
            "profile": profile(rows)}


def profile(rows, step=5):
    """Knuth por niveles: N(j) = E[2^(ramas a profundidad < j) * 1(L >= j)].

    Devuelve [j, N(j), error_estandar, caminos que alcanzan j] cada `step`.
    """
    m = len(rows)
    top = max(r["dec"] for r in rows)
    out = []
    for j in range(0, top + 1, step):
        xs = [2.0 ** sum(b < j for b in r["branches"]) if r["dec"] >= j else 0.0
              for r in rows]
        mu = sum(xs) / m
        se = statistics.stdev(xs) / math.sqrt(m) if m > 1 else float("nan")
        out.append([j, mu, se, sum(r["dec"] >= j for r in rows)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("s", type=int)
    ap.add_argument("t", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("d", type=int, help="grado rojo de la raiz")
    ap.add_argument("--window", type=int, nargs=2, required=True)
    ap.add_argument("--k", type=int, required=True, help="decisiones por cubo")
    ap.add_argument("--samples", type=int, default=30, help="-1 = arbol exacto")
    ap.add_argument("--population", type=int,
                    help="modo poblacion con P nodos por nivel (ignora --samples)")
    ap.add_argument("--leaf-time", type=float, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    tag = f"r{a.s}{a.t}_{a.n}_d{a.d}_w{a.window[0]}-{a.window[1]}"
    cnf = os.path.join(WORK, f"est_{tag}.cnf")
    if not os.path.exists(cnf):
        rc.write(cnf, a.s, a.t, a.n, root_degree=a.d, window=tuple(a.window),
                 lex="root")
    order = os.path.join(WORK, f"est_{tag}.order")
    with open(order, "w") as f:
        f.write(" ".join(map(str, edge_order(a.n, a.d))) + "\n")
    m = -a.population if a.population else a.samples
    out = run([ESTIMATOR, cnf, order, str(a.k), str(m),
               str(a.leaf_time), str(a.seed)])
    if out.returncode != 0:
        sys.exit(out.stderr)
    rows, levels = [], []
    for line in out.stdout.split("\n"):
        if line.startswith("L "):
            _, j, lw, cnt = line.split()
            levels.append([int(j), float(lw), int(cnt)])
        elif line.strip():
            lw, dec, res, secs, confl, br = line.split()
            rows.append({"log2w": float(lw), "dec": int(dec), "res": res,
                         "secs": float(secs), "conflicts": int(confl),
                         "branches": [] if br == "-" else list(map(int, br.split(",")))})
    if a.population:
        w = [2.0 ** r["log2w"] for r in rows]
        t = [a.leaf_time if r["res"] == "TIMEOUT" else r["secs"] for r in rows]
        solved = [r["secs"] for r in rows if r["res"] in ("UNSAT", "SAT")]
        summ = {"population": a.population, "levels": levels,
                "leaves_est": sum(w), "cost_s_est": sum(x * y for x, y in zip(w, t)),
                "cost_is_lower_bound": any(r["res"] == "TIMEOUT" for r in rows),
                "mean_leaf_s": sum(t) / len(t) if t else None,
                "mean_leaf_s_solved": sum(solved) / len(solved) if solved else None,
                "outcomes": {k: sum(r["res"] == k for r in rows)
                             for k in ("UNSAT", "SAT", "TIMEOUT")}}
    elif a.samples < 0:  # arbol exacto: peso 1 por hoja
        summ = {"exact": True, "leaves": len(rows),
                "cost_s": sum(a.leaf_time if r["res"] == "TIMEOUT" else r["secs"]
                              for r in rows),
                "outcomes": {k: sum(r["res"] == k for r in rows)
                             for k in ("UNSAT", "SAT", "TIMEOUT")}}
    else:
        summ = summarize(rows, a.leaf_time)
    rec = {"s": a.s, "t": a.t, "n": a.n, "d": a.d, "window": a.window,
           "k": a.k, "leaf_time": a.leaf_time, "seed": a.seed, **summ}
    os.makedirs(os.path.join(RESULTS, "estimates"), exist_ok=True)
    mode = (f"pop{a.population}" if a.population else
            "exact" if a.samples < 0 else f"n{a.samples}")
    name = f"{tag}_k{a.k}_{mode}_T{a.leaf_time:g}_s{a.seed}.json"
    with open(os.path.join(RESULTS, "estimates", name), "w") as f:
        json.dump({**rec, "rows": rows}, f, indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
