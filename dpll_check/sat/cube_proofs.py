#!/usr/bin/env python3
"""Resuelve cubos sueltos con prueba LRAT y la comprueba (mide tamano por cubo).

Uso: python cube_proofs.py <cnf> <cubos.txt> [--max N] [--timeout S]
Cada cubo se anade como clausulas unitarias a una copia de la CNF; la prueba
de esa copia es la refutacion certificada del cubo.
"""
import argparse
import json
import os
import sys

from certify import solve
from common import RESULTS, WORK


def with_units(cnf, cube, out):
    with open(cnf) as f, open(out, "w", newline="\n") as g:
        for line in f:
            if line.startswith("p cnf"):
                _, _, nv, nc = line.split()
                g.write(f"p cnf {nv} {int(nc) + len(cube)}\n")
                break
            g.write(line)
        for chunk in iter(lambda: f.read(1 << 22), ""):
            g.write(chunk)
        g.writelines(f"{l} 0\n" for l in cube)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf")
    ap.add_argument("cubes")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=os.path.join(RESULTS, "cube_proofs.json"))
    a = ap.parse_args()
    cubes = [list(map(int, ln.split()))[:-1] for ln in open(a.cubes) if ln.strip()]
    recs = []
    for i, cube in enumerate(cubes[:a.max]):
        path = os.path.join(WORK, f"cube_{i}.cnf")
        with_units(a.cnf, cube, path)
        rec = solve(path, a.timeout)
        rec.pop("model", None)
        rec.update(cube_len=len(cube))
        os.remove(path)
        recs.append(rec)
        print(json.dumps(rec), flush=True)
    with open(a.out, "w") as f:
        json.dump({"cnf": os.path.basename(a.cnf), "cubes": recs}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
