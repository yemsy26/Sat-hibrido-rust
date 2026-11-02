#!/usr/bin/env python3
"""Resuelve (s,t,n) con CaDiCaL y certifica cada veredicto.

UNSAT: prueba LRAT de CaDiCaL comprobada con lrat-check; ademas
lrat-check reescribe los lemas como DRAT y drat-trim los comprueba
otra vez (dos verificadores independientes sobre la misma CNF).
SAT:   el modelo se comprueba con ramsey_cnf.check_coloring, que busca
cliques monocromaticos directamente sobre la coloracion, sin la CNF.

Modos:
  plain    solo la formula F (sin simetria ni clases).
  classes  particion en clases de raiz (common.root_classes) con lex
           dentro de cada vecindario; no usa ningun lema externo.
  lemma    como classes, pero restringe los grados con numeros de
           Ramsey menores (LEMA externo: ver README).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import ramsey_cnf as rc
from common import (CADICAL, DRAT_TRIM, LRAT_CHECK, RESULTS, WORK,
                    root_classes, run)

# Valores conocidos usados SOLO por el modo lemma y para el veredicto esperado.
RAMSEY = {(3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18, (3, 7): 23,
          (3, 8): 28, (3, 9): 36, (4, 4): 18, (4, 5): 25}


def ramsey(s, t):
    if min(s, t) <= 1:
        return 1
    if min(s, t) == 2:
        return max(s, t)
    return RAMSEY[(min(s, t), max(s, t))]


def lemma_range(s, t, n):
    """Grado rojo de todo vertice en [n - R(s,t-1), R(s-1,t) - 1]."""
    return (max(0, n - ramsey(s, t - 1)), min(n - 1, ramsey(s - 1, t) - 1))


def solve(cnf, timeout, proof=True, check=True):
    rec = {"cnf": os.path.basename(cnf)}
    lrat = cnf[:-4] + ".lrat"
    cmd = [CADICAL, "-q"]  # el binario MinGW no admite -t: se usa timeout
    if proof:
        cmd += ["--lrat", "--no-binary", cnf, lrat]
    else:
        cmd += [cnf]
    t0 = time.perf_counter()
    try:
        out = run(cmd, timeout=timeout)
        code = out.returncode
    except subprocess.TimeoutExpired:
        code = None
    rec["solve_s"] = round(time.perf_counter() - t0, 3)
    rec["result"] = {10: "SAT", 20: "UNSAT", None: "TIMEOUT"}.get(code, "UNKNOWN")
    if code == 10:
        rec["model"] = {int(x) for line in out.stdout.splitlines()
                        if line.startswith("v ") for x in line[2:].split()
                        if int(x) > 0}
    if code == 20 and proof:
        rec["proof_bytes"] = os.path.getsize(lrat)
        with open(lrat) as f:
            rec["proof_lemmas"] = sum(1 for ln in f if " d " not in ln)
        if check:
            drat = cnf[:-4] + ".drat"
            t0 = time.perf_counter()
            o = run([LRAT_CHECK, cnf, lrat, drat])
            rec["lrat_check_s"] = round(time.perf_counter() - t0, 3)
            rec["lrat_verified"] = "c VERIFIED" in o.stdout.splitlines()
            t0 = time.perf_counter()
            o = run([DRAT_TRIM, cnf, drat, "-t", "100000"])
            rec["drat_trim_s"] = round(time.perf_counter() - t0, 3)
            rec["drat_verified"] = "s VERIFIED" in o.stdout.splitlines()
            os.remove(drat)
        os.remove(lrat)
    elif proof and os.path.exists(lrat):
        os.remove(lrat)
    return rec


def certify(s, t, n, mode, timeout, keep_cnf=False, stop_on_sat=False,
            only=None, proof=True):
    os.makedirs(WORK, exist_ok=True)
    if mode == "plain":
        jobs = [(None, None, "none")]
    else:
        rng = lemma_range(s, t, n) if mode == "lemma" else None
        jobs = [(d, w, "root") for d, w in root_classes(s, t, n, rng)
                if only is None or d in only]
    recs = []
    for d, w, lex in jobs:
        path = os.path.join(WORK, f"r{s}{t}_{n}_{mode}_{d}.cnf")
        nv, nc = rc.write(path, s, t, n, root_degree=d, window=w, lex=lex)
        rec = solve(path, timeout, proof=proof)
        rec.update(root_degree=d, window=w, vars=nv, clauses=nc)
        if rec["result"] == "SAT":
            red = rc.coloring_from_model(n, rec.pop("model"))
            rec["model_errors"] = rc.check_coloring(s, t, n, red, d, w)
            rec["model_ok"] = not rec["model_errors"]
            rec["red_degrees"] = sorted(sum(r) for r in red)
        if not keep_cnf:
            os.remove(path)
        recs.append(rec)
        print(f"  ({s},{t},{n}) {mode} d={d} w={w}: {rec['result']} "
              f"{rec['solve_s']}s lrat={rec.get('lrat_verified')} "
              f"drat={rec.get('drat_verified')} model_ok={rec.get('model_ok')}",
              flush=True)
        if stop_on_sat and rec["result"] == "SAT":
            break
    if any(r["result"] == "SAT" and r.get("model_ok") for r in recs):
        verdict = "SAT"
    elif recs and all(r["result"] == "UNSAT" and r.get("lrat_verified")
                      and r.get("drat_verified") for r in recs):
        verdict = "UNSAT"
    elif not recs:
        verdict = "UNSAT"  # el lema deja cero clases: no hay prueba SAT que hacer
    elif all(r["result"] == "UNSAT" for r in recs):
        verdict = "UNSAT-sin-prueba"  # --no-proof: solo dato de tiempo
    else:
        verdict = "UNKNOWN"
    if only is not None and verdict == "UNSAT":
        verdict = "UNSAT-parcial"  # faltan clases: no es una demostracion
    return {"s": s, "t": t, "n": n, "mode": mode, "verdict": verdict,
            "by_lemma_only": not recs,
            "expected": "UNSAT" if n >= ramsey(s, t) else "SAT",
            "total_solve_s": round(sum(r["solve_s"] for r in recs), 3),
            "total_proof_bytes": sum(r.get("proof_bytes", 0) for r in recs),
            "total_lemmas": sum(r.get("proof_lemmas", 0) for r in recs),
            "classes": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="s,t,n (por defecto: validacion)")
    ap.add_argument("--modes", default="classes,plain")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=os.path.join(RESULTS, "validation.json"))
    ap.add_argument("--only", help="solo estas clases d (coma); el veredicto"
                    " global no es valido si se omiten clases")
    ap.add_argument("--no-proof", action="store_true",
                    help="sin LRAT (solo mide tiempo; no certifica)")
    a = ap.parse_args()
    cases = ([tuple(map(int, c.split(","))) for c in a.cases] or
             [(3, 3, 5), (3, 3, 6), (3, 4, 8), (3, 4, 9), (3, 5, 13),
              (3, 5, 14), (4, 4, 17), (4, 4, 18)])
    out = []
    for s, t, n in cases:
        for mode in a.modes.split(","):
            only = set(map(int, a.only.split(","))) if a.only else None
            r = certify(s, t, n, mode, a.timeout, only=only,
                        proof=not a.no_proof)
            ok = r["verdict"] == r["expected"]
            print(f"({s},{t},{n}) {mode:7s} veredicto={r['verdict']:7s} "
                  f"esperado={r['expected']:5s} {'OK' if ok else 'NO'} "
                  f"t={r['total_solve_s']}s prueba={r['total_proof_bytes']}B"
                  f"{' (solo lema, 0 clases)' if r['by_lemma_only'] else ''}",
                  flush=True)
            out.append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    wrong = [r for r in out if r["verdict"] in ("SAT", "UNSAT")
             and r["verdict"] != r["expected"]]
    open_ = [r for r in out if r["verdict"] not in ("SAT", "UNSAT")]
    print(f"incorrectos={len(wrong)} sin_decidir={len(open_)} "
          f"correctos_certificados={len(out) - len(wrong) - len(open_)}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
