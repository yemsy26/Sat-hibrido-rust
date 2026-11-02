#!/usr/bin/env python3
"""Tablas Markdown del README a partir de results/*.json."""
import glob
import json
import math
import os
import sys

from common import RESULTS


def cases_table(paths):
    print("| caso | modo | clases | veredicto | esperado | t solver | prueba LRAT | lrat-check | drat-trim |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        for r in json.load(open(p)):
            cl = r["classes"]
            uns = [c for c in cl if c["result"] == "UNSAT"]
            lr = f"{sum(c.get('lrat_verified', False) for c in uns)}/{len(uns)}" if uns else "–"
            dr = f"{sum(c.get('drat_verified', False) for c in uns)}/{len(uns)}" if uns else "–"
            tchk = sum(c.get("lrat_check_s", 0) + c.get("drat_trim_s", 0) for c in uns)
            note = " (lema: 0 clases)" if r.get("by_lemma_only") else ""
            timeouts = sum(c["result"] == "TIMEOUT" for c in cl)
            verdict = r["verdict"] + (f" ({timeouts} TIMEOUT)" if timeouts else "")
            print(f"| ({r['s']},{r['t']},{r['n']}) | {r['mode']}{note} | {len(cl)} | "
                  f"{verdict} | {r['expected']} | {r['total_solve_s']:.2f} s | "
                  f"{r['total_proof_bytes'] / 1e6:.2f} MB | {lr} | {dr} ({tchk:.1f} s) |")


def k43_table():
    print("| d | ventana | P | prof. máx. viva | log10 N(153) | log10 N(429) | log10 Σ_j N(j) |")
    print("|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(os.path.join(RESULTS, "estimates", "r55_43_*_k861_pop*.json"))):
        r = json.load(open(p))
        lv = {j: lw for j, lw, _ in r["levels"] if lw >= 0}
        top = max(lv)
        tot = math.log10(sum(2.0 ** x for x in lv.values()))
        g = lambda j: f"{lv[j] * math.log10(2):.1f}" if j in lv else "extinta"
        print(f"| {r['d']} | {r['window']} | {r['population']} | {top} | {g(153)} | {g(429)} | {tot:.1f} |")


def leaves_table():
    print("| d | j (aristas fijadas) | muestreo | T hoja | log10 N(j) | UNSAT / TIMEOUT | media resueltas | log10 coste (s) |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "estimates", "r55_43_*.json")):
        r = json.load(open(p))
        if r["k"] == 861 or r["leaf_time"] < 1 or not r["leaves_est"]:
            continue
        rows.append(r)
    for r in sorted(rows, key=lambda r: (r["d"], r["k"], r["leaf_time"], r["seed"])):
        o = r["outcomes"]
        lb = "≥ " if r["cost_is_lower_bound"] else "≈ "
        ms = r["mean_leaf_s_solved"]
        how = f"población P={r['population']}" if "population" in r else f"Knuth {r['samples']} caminos"
        print(f"| {r['d']} | {r['k']} | {how} (s{r['seed']}) | {r['leaf_time']:g} s | "
              f"{math.log10(r['leaves_est']):.1f} | {o['UNSAT']} / {o['TIMEOUT']} | "
              f"{f'{ms:.2f} s' if ms is not None else '–'} | {lb}{math.log10(r['cost_s_est']):.1f} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "cases"):
        cases_table(sys.argv[2:] or [os.path.join(RESULTS, "validation.json")])
    if what in ("all", "k43"):
        k43_table()
        leaves_table()
