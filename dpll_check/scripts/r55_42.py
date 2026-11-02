"""Reproduce McKay-Radziszowski: ningun grafo Ramsey(5,5,42) conocido extiende a K43.

El grafo de 42 vertices ocupa los vertices 1..42 de K43 (arista = rojo, no
arista = azul); las 42 aristas del vertice 0 quedan libres. Ejecutar siempre
con `python -I`: el archivo graph6 son datos externos no confiables.

  fetch    descarga r55_42some.g6 de la pagina de B. McKay y verifica SHA-256
  convert  graph6 -> 656 bases DIMACS (328 grafos + sus 328 complementos)
  audit    compila scripts/r55_check_bases.rs: check_reference del motor por base
  gpu      gpu_ramsey43 --lookahead --explicit-trail --epochs 0 por base
  cpu      DPLL independiente en CPU sobre la estrella del vertice 0
  summary  tabla Markdown a partir de los CSV
  all      convert, audit, gpu, cpu y summary (no descarga ni compila el motor)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
URL = "https://users.cecs.anu.edu.au/~bdm/data/r55_42some.g6"
SHA256 = "067902e853d87b49bcef0d1d4c0e3bbadd238ee18bc65341b079a3ca4780eccb"
G6 = ROOT / "data" / "r55" / "r55_42some.g6"
WORK = ROOT / "target" / "r55_42"
BASES = WORK / "bases"
CHECKER = WORK / "r55_check_bases.exe"
ENGINE = ROOT / "target" / "neuro-trt" / "release" / "gpu_ramsey43.exe"
RESULTS = ROOT / "results" / "r55_42"
GPU_ARGS = ["--lookahead", "--explicit-trail", "--epochs", "0"]
N, K43 = 42, 43


def edge_id(a: int, b: int) -> int:
    """Igual que `edge_id` de src/ramsey43_gpu.rs: (0,1)=1, ..., (41,42)=903."""
    assert 0 <= a < b < K43
    return a * (2 * K43 - a - 1) // 2 + b - a


def decode_graph6(line: str) -> list[int]:
    """Filas de adyacencia como mascaras; solo n <= 62 (cabecera de un byte)."""
    data = [ord(ch) - 63 for ch in line.strip()]
    if not data or not all(0 <= x < 64 for x in data) or data[0] > 62:
        raise ValueError("graph6 invalido o n > 62")
    n, bits = data[0], data[1:]
    if len(bits) != -(-n * (n - 1) // 2 // 6):
        raise ValueError(f"longitud graph6 incorrecta para n={n}")
    rows = [0] * n
    k = 0
    for j in range(1, n):  # graph6: triangulo superior por columnas
        for i in range(j):
            if bits[k // 6] >> (5 - k % 6) & 1:
                rows[i] |= 1 << j
                rows[j] |= 1 << i
            k += 1
    if any(bits[k // 6] >> (5 - k % 6) & 1 for k in range(k, 6 * len(bits))):
        raise ValueError("relleno graph6 no nulo")
    return rows


def has_clique(rows: list[int], cand: int, k: int) -> bool:
    if k == 0:
        return True
    while cand.bit_count() >= k:
        v = (cand & -cand).bit_length() - 1
        cand &= cand - 1
        if has_clique(rows, cand & rows[v], k - 1):
            return True
    return False


def complement(rows: list[int], n: int) -> list[int]:
    full = (1 << n) - 1
    return [full & ~row & ~(1 << v) for v, row in enumerate(rows)]


def load_graphs(path: Path) -> list[tuple[str, list[int]]]:
    """328 grafos del archivo y sus complementos: g001..g328, g001c..g328c."""
    lines = [line for line in path.read_text(encoding="ascii").splitlines() if line]
    graphs = [(f"g{i:03d}", decode_graph6(line)) for i, line in enumerate(lines, 1)]
    return graphs + [(name + "c", complement(rows, N)) for name, rows in graphs]


def base_text(name: str, rows: list[int]) -> str:
    literals = [
        edge_id(i + 1, j + 1) * (1 if rows[i] >> j & 1 else -1)
        for i in range(N) for j in range(i + 1, N)
    ]
    return (
        f"c r55_42 {name}: Ramsey(5,5,42) en vertices 1..42 de K43; rojo=arista, azul=no arista\n"
        f"c estrella del vertice 0 (variables 1..42) libre\n"
        f"v {' '.join(map(str, literals))} 0\n"
    )


def cmd_fetch(args: argparse.Namespace) -> None:
    import urllib.request

    args.g6.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        sys.exit("SHA-256 distinto del registrado; no se guarda")
    args.g6.write_bytes(data)
    print(f"c {args.g6}: {len(data)} bytes, sha256 ok")


def cmd_convert(args: argparse.Namespace) -> None:
    if hashlib.sha256(args.g6.read_bytes()).hexdigest() != SHA256:
        sys.exit(f"{args.g6}: SHA-256 distinto de {SHA256}")
    graphs = load_graphs(args.g6)
    if len(graphs) != 656 or len({tuple(r) for _, r in graphs}) != 656:
        sys.exit(f"se esperaban 656 grafos distintos, hay {len(graphs)}")
    args.bases.mkdir(parents=True, exist_ok=True)
    for name, rows in graphs:
        full = (1 << N) - 1
        if has_clique(rows, full, 5) or has_clique(complement(rows, N), full, 5):
            sys.exit(f"{name}: contiene K5 o conjunto independiente de 5; decodificacion erronea")
        (args.bases / f"{name}.sat").write_text(base_text(name, rows), encoding="ascii")
    print(f"c {len(graphs)} bases en {args.bases}; sin K5 ni I5 en Python")


def cmd_audit(args: argparse.Namespace) -> None:
    args.results.mkdir(parents=True, exist_ok=True)
    source = ROOT / "scripts" / "r55_check_bases.rs"
    CHECKER.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rustc", "--edition", "2021", "-O", str(source), "-o", str(CHECKER)], check=True)
    run = subprocess.run([str(CHECKER), str(args.bases)], capture_output=True, text=True)
    (args.results / "audit.csv").write_text(run.stdout, encoding="utf-8")
    print(run.stderr.strip())
    if run.returncode:
        sys.exit("auditoria check_reference fallida; ver audit.csv")


def engine_running() -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq gpu_ramsey43.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True,
    ).stdout
    return "gpu_ramsey43.exe" in out


GPU_FIELDS = ["base", "verdict", "elapsed_ms", "transitions", "prunes", "completed",
              "decisions", "forced_assignments", "lookahead_conflicts", "launches", "wall_s"]


def cmd_gpu(args: argparse.Namespace) -> None:
    if engine_running():
        sys.exit("ya hay un gpu_ramsey43.exe activo (asedio u otra prueba); no se comparte la GPU")
    if not args.engine.is_file():
        sys.exit(f"falta {args.engine}; compilar con --features neuro-trt --target-dir target/neuro-trt")
    args.results.mkdir(parents=True, exist_ok=True)
    logs = WORK / "gpu_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.bases.glob("*.sat"))[: args.limit or None]:
        started = time.perf_counter()
        try:
            run = subprocess.run([str(args.engine), "--base", str(path), *GPU_ARGS],
                                 capture_output=True, text=True, timeout=args.timeout)
            out, code = run.stdout, run.returncode
        except subprocess.TimeoutExpired as error:
            out, code = error.stdout or "", "timeout"
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
        wall = time.perf_counter() - started
        (logs / f"{path.stem}.log").write_text(out + f"c exit={code}\n", encoding="utf-8")
        verdict = next((line[2:] for line in out.splitlines() if line.startswith("s ")), f"ERROR({code})")
        header = next((i for i, line in enumerate(out.splitlines()) if line.startswith("elapsed_ms,")), None)
        stats = {}
        if header is not None:
            lines = out.splitlines()
            stats = dict(zip(lines[header].split(","), lines[header + 1].split(",")))
        if "c base: free_edges=42;" not in out:
            verdict = f"ERROR(base {verdict})"
        row = {field: stats.get(field, "") for field in GPU_FIELDS}
        row.update(base=path.stem, verdict=verdict, wall_s=f"{wall:.3f}")
        rows.append(row)
        print(f"{path.stem}: {verdict} {row['elapsed_ms']} ms, {row['transitions']} transiciones", flush=True)
    write_csv(args.results / "gpu.csv", GPU_FIELDS, rows)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --- Comprobacion independiente en CPU -------------------------------------
# Toda K5 monocromatica de la extension esta dentro de 1..42 (excluido por la
# auditoria) o es {0} + un K4 monocromatico Q de 1..42 cuyas aristas a 0 tienen
# su color. Con x_v = color rojo de (0,v): cada K4 rojo prohibe Q todo rojo y
# cada K4 azul prohibe Q todo azul. Es un problema de 42 variables, ajeno a las
# 903 variables, al orden de ramificacion y a las podas de grado del motor.

def read_base(path: Path) -> tuple[list[int], list[int]]:
    """Filas roja/azul de K43 desde la base; numeracion por enumeracion de pares."""
    pairs = [(a, b) for a in range(K43) for b in range(a + 1, K43)]
    red, blue = [0] * K43, [0] * K43
    tokens = [t for line in path.read_text(encoding="ascii").splitlines()
              if line.startswith("v ") for t in line.split()[1:]]
    if tokens[-1] != "0":
        raise ValueError(f"{path}: falta terminador 0")
    for literal in map(int, tokens[:-1]):
        a, b = pairs[abs(literal) - 1]
        rows = red if literal > 0 else blue
        rows[a] |= 1 << b
        rows[b] |= 1 << a
    return red, blue


def mono_k4(rows: list[int], vertices: int) -> list[int]:
    found = []

    def grow(clique: int, cand: int, size: int) -> None:
        if size == 4:
            found.append(clique)
            return
        while cand:
            v = (cand & -cand).bit_length() - 1
            cand &= cand - 1
            grow(clique | 1 << v, cand & rows[v], size + 1)

    grow(0, vertices, 0)
    return found


def dpll(no_all_true: list[int], no_all_false: list[int], free: int) -> tuple[int | None, int]:
    """Busca T (rojos) con ningun Q de no_all_true dentro de T ni de no_all_false fuera."""
    nodes = 0

    def propagate(t: int, f: int) -> tuple[int, int] | None:
        changed = True
        while changed:
            changed = False
            for q in no_all_true:
                if not q & f:
                    rest = q & ~t
                    if not rest:
                        return None
                    if not rest & (rest - 1):
                        f |= rest
                        changed = True
            for q in no_all_false:
                if not q & t:
                    rest = q & ~f
                    if not rest:
                        return None
                    if not rest & (rest - 1):
                        t |= rest
                        changed = True
        return t, f

    def search(t: int, f: int) -> int | None:
        nonlocal nodes
        nodes += 1
        state = propagate(t, f)
        if state is None:
            return None
        t, f = state
        unassigned = free & ~(t | f)
        if not unassigned:
            return t
        score = {}
        for clauses, sat in ((no_all_true, f), (no_all_false, t)):
            for q in clauses:
                if not q & sat:
                    rest = q & unassigned
                    weight = 16 >> rest.bit_count()
                    while rest:
                        v = rest & -rest
                        score[v] = score.get(v, 0) + weight
                        rest ^= v
        v = max(score, key=score.get) if score else unassigned & -unassigned
        model = search(t | v, f)
        return model if model is not None else search(t, f | v)

    return search(0, 0), nodes


def extension_problem(red: list[int], blue: list[int], vertices: int):
    return mono_k4(red, vertices), mono_k4(blue, vertices)


def verify_extension(red: list[int], blue: list[int], vertices: int, t: int) -> bool:
    """Completa la estrella de 0 sobre `vertices` y busca K5 monocromatico."""
    red, blue = red[:], blue[:]
    red[0], blue[0] = t, vertices & ~t
    for v in range(1, K43):
        if vertices >> v & 1:
            (red if t >> v & 1 else blue)[v] |= 1
    scope = vertices | 1
    return not any(has_clique(rows, scope, 5) for rows in (red, blue))


CPU_FIELDS = ["base", "verdict", "red_k4", "blue_k4", "dpll_nodes", "cpu_s",
              "control_sat", "control_nodes"]


def cmd_cpu(args: argparse.Namespace) -> None:
    args.results.mkdir(parents=True, exist_ok=True)
    all_vertices = ((1 << K43) - 1) & ~1
    rows = []
    for path in sorted(args.bases.glob("*.sat"))[:: args.every]:
        red, blue = read_base(path)
        if red[0] | blue[0] or any(has_clique(r, all_vertices, 5) for r in (red, blue)):
            sys.exit(f"{path.stem}: base invalida para la comprobacion CPU")
        started = time.perf_counter()
        red4, blue4 = extension_problem(red, blue, all_vertices)
        model, nodes = dpll(red4, blue4, all_vertices)
        if model is not None and not verify_extension(red, blue, all_vertices, model):
            sys.exit(f"{path.stem}: modelo CPU con K5; DPLL defectuoso")
        verdict = "UNSATISFIABLE" if model is None else "SATISFIABLE"
        # Control positivo: sin el vertice 42, la vecindad de 42 extiende el resto.
        sub = all_vertices & ~(1 << 42)
        c_model, c_nodes = dpll(*extension_problem(red, blue, sub), sub)
        control = c_model is not None and verify_extension(red, blue, sub, c_model)
        rows.append(dict(base=path.stem, verdict=verdict, red_k4=len(red4), blue_k4=len(blue4),
                         dpll_nodes=nodes, cpu_s=f"{time.perf_counter() - started:.3f}",
                         control_sat=control, control_nodes=c_nodes))
        print(f"{path.stem}: {verdict}, K4 rojo/azul {len(red4)}/{len(blue4)}, "
              f"{nodes} nodos; control SAT={control}", flush=True)
    write_csv(args.results / "cpu.csv", CPU_FIELDS, rows)


def cmd_summary(args: argparse.Namespace) -> None:
    def read(name: str) -> list[dict]:
        path = args.results / name
        return list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []

    audit, gpu, cpu = read("audit.csv"), read("gpu.csv"), read("cpu.csv")
    gpu_unsat = [r for r in gpu if r["verdict"] == "UNSATISFIABLE"]
    cpu_unsat = [r for r in cpu if r["verdict"] == "UNSATISFIABLE"]
    exceptions = [r for r in gpu if r["verdict"] != "UNSATISFIABLE"]
    exceptions += [r for r in cpu if r["verdict"] != "UNSATISFIABLE" or r["control_sat"] != "True"]
    gpu_cpu = {r["base"]: r["verdict"] for r in cpu}
    disagree = [r["base"] for r in gpu if r["base"] in gpu_cpu and gpu_cpu[r["base"]] != r["verdict"]]

    def stats(rows: list[dict], key: str) -> str:
        values = sorted(float(r[key]) for r in rows if r[key])
        if not values:
            return "-"
        return f"{sum(values):.0f} / {values[len(values) // 2]:.1f} / {values[-1]:.1f}"

    lines = [
        "# Extensiones de los 656 grafos Ramsey(5,5,42) conocidos a K43",
        "",
        f"Datos: `{URL}` (SHA-256 `{SHA256}`), 328 grafos y sus complementos.",
        f"Motor: `gpu_ramsey43 --base <g>.sat {' '.join(GPU_ARGS)}`; reproducir con "
        "`python -I scripts/r55_42.py all`.",
        "",
        "| Comprobación | Bases | Resultado |",
        "|---|---|---|",
        f"| `check_reference` sin K5 en la parte fija, estrella de 0 libre | {len(audit)} | "
        f"{sum(r['ok'] == 'true' for r in audit)} correctas |",
        f"| GPU `s UNSATISFIABLE` | {len(gpu)} | {len(gpu_unsat)}/{len(gpu)} |",
        f"| CPU DPLL independiente UNSAT | {len(cpu)} | {len(cpu_unsat)}/{len(cpu)} |",
        f"| CPU control positivo (sin vértice 42 extiende) | {len(cpu)} | "
        f"{sum(r['control_sat'] == 'True' for r in cpu)}/{len(cpu)} |",
        f"| Desacuerdos GPU/CPU | {len(disagree)} | {', '.join(disagree) or 'ninguno'} |",
        "",
        "| Métrica GPU (total / mediana / máx.) | Valor |",
        "|---|---|",
        f"| elapsed_ms del motor | {stats(gpu, 'elapsed_ms')} |",
        f"| transiciones | {stats(gpu, 'transitions')} |",
        f"| decisiones | {stats(gpu, 'decisions')} |",
        f"| tiempo de pared por proceso (s) | {stats(gpu, 'wall_s')} |",
        f"| nodos DPLL en CPU | {stats(cpu, 'dpll_nodes')} |",
        "",
        "Excepciones: " + (", ".join(f"{r['base']} ({r['verdict']})" for r in exceptions) or "ninguna."),
        "",
        "Detalle por grafo: `gpu.csv`, `cpu.csv`, `audit.csv`.",
    ]
    (args.results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "convert", "audit", "gpu", "cpu", "summary", "all"])
    parser.add_argument("--g6", type=Path, default=G6)
    parser.add_argument("--bases", type=Path, default=BASES)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--timeout", type=float, default=3600, help="segundos por base en GPU")
    parser.add_argument("--limit", type=int, default=0, help="solo las primeras N bases en GPU")
    parser.add_argument("--every", type=int, default=1, help="CPU: una de cada N bases")
    args = parser.parse_args()
    steps = ["convert", "audit", "gpu", "cpu", "summary"] if args.command == "all" else [args.command]
    for step in steps:
        globals()[f"cmd_{step}"](args)


if __name__ == "__main__":
    main()
