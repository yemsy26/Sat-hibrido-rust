"""Compara motor con CaDiCaL y Kissat en las mismas instancias, con el mismo limite externo.

Ejecutar con `python -I`: las instancias descargadas son datos externos no confiables.

  tools     compila Kissat (fuera de esta carpeta: su configure no admite espacios en la ruta)
  fetch     descarga una muestra fija de la SAT Competition 2023 (pista main) a data/bench/sc2023
  generate  genera las familias propias en data/bench/familias
  run       ejecuta los solvers; los resultados se acumulan en results/comparacion/<conjunto>.jsonl
  certify   mide lo que cuesta certificar los UNSAT resueltos: motor+VeriPB y CaDiCaL+drat-trim
  report    escribe results/comparacion_solvers.md

  python -I scripts/comparar_solvers.py tools
  python -I scripts/comparar_solvers.py fetch
  python -I scripts/comparar_solvers.py generate
  python -I scripts/comparar_solvers.py run sc2023 --limit 120 --jobs 4
  python -I scripts/comparar_solvers.py run familias --limit 300 --jobs 4
  python -I scripts/comparar_solvers.py certify --limit 600 --jobs 4
  python -I scripts/comparar_solvers.py report

Cada modelo se evalua aqui contra la CNF, sin creer a ningun solver. Dos veredictos opuestos en una
misma instancia se marcan en el informe: uno de los solvers tiene un error.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import lzma
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "bench"
OUT = ROOT / "results" / "comparacion"
REPORT = ROOT / "results" / "comparacion_solvers.md"
EXE = ".exe" if os.name == "nt" else ""
MOTOR = ROOT / "target" / "release" / f"motor{EXE}"
CADICAL = ROOT / "sat" / "tools" / "cadical" / "build" / f"cadical{EXE}"
DRAT_TRIM = ROOT / "sat" / "tools" / "drat-trim" / f"drat-trim{EXE}"
KISSAT_TAG = "rel-4.0.4"
KISSAT_DIR = Path(tempfile.gettempdir()) / "motor-bench" / "kissat"
KISSAT = KISSAT_DIR / "build" / f"kissat{EXE}"
GBD_TRACK = "https://benchmark-database.de/getinstances?track=main_2023"
# Muestra: las de hasta 1 MiB comprimidas, por hash, a intervalos regulares.
SAMPLE_CAP = 1 << 20
SAMPLE_SIZE = 60
PAR_FACTOR = 2


def solvers(extra: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    return {
        "kissat": [str(KISSAT)],
        "cadical": [str(CADICAL), "-q"],
        "motor": [str(MOTOR), "solve", "{cnf}", "--no-verify"],
        "motor-4h": [str(MOTOR), "solve", "{cnf}", "--no-verify", "--workers", "4"],
    }


def command_tools(_: argparse.Namespace) -> None:
    if not KISSAT.exists():
        KISSAT_DIR.parent.mkdir(parents=True, exist_ok=True)
        if not KISSAT_DIR.exists():
            subprocess.run(["git", "clone", "-q", "-c", "advice.detachedHead=false", "--depth", "1", "--branch",
                            KISSAT_TAG, "https://github.com/arminbiere/kissat", str(KISSAT_DIR)], check=True)
        compiler = "gcc"
        arguments = []
        if os.name == "nt":
            # Kissat no contempla Windows: sin esto, MinGW usa la disposicion de campos de bits de
            # MSVC, el tipo `watch` deja de ocupar 32 bits y se cae al propagar. alarm y SIGALRM
            # solo los usa --time, y el limite lo pone este banco desde fuera. getrusage no existe:
            # --quiet quita las estadisticas que lo usan.
            shim = KISSAT_DIR / "winshim.h"
            shim.write_text("#define SIGALRM 14\nstatic inline unsigned alarm (unsigned s) { (void) s; return 0; }\n")
            compiler = f"gcc -mno-ms-bitfields -include {shim.as_posix()}"
            arguments = ["--quiet"]
        bash = shutil.which("bash") or "bash"
        subprocess.run([bash, "-c", f'CC="{compiler}" ./configure {" ".join(arguments)} && make -C build -j8 kissat'],
                       cwd=KISSAT_DIR, check=True, stdout=subprocess.DEVNULL)
    print(KISSAT, subprocess.run([str(KISSAT), "--version"], capture_output=True, text=True).stdout.strip())
    if not CADICAL.exists():
        sys.exit("Falta CaDiCaL: bash sat/build_tools.sh")


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def command_fetch(_: argparse.Namespace) -> None:
    folder = BENCH / "sc2023"
    folder.mkdir(parents=True, exist_ok=True)
    urls = sorted(line.strip().replace("http://", "https://") for line in download(GBD_TRACK).decode().splitlines() if line.strip())

    def size(url: str) -> tuple[str, int]:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return url, int(response.headers["Content-Length"])

    with futures.ThreadPoolExecutor(16) as pool:
        small = [url for url, bytes_ in pool.map(size, urls) if bytes_ <= SAMPLE_CAP]
    step = len(small) / SAMPLE_SIZE
    chosen = [small[int(index * step)] for index in range(SAMPLE_SIZE)]
    print(f"{len(urls)} instancias, {len(small)} de hasta {SAMPLE_CAP} bytes, {len(chosen)} elegidas")

    def fetch(url: str) -> None:
        target = folder / (url.rsplit("/", 1)[1] + ".cnf")
        if not target.exists():
            target.write_bytes(lzma.decompress(download(url)))

    with futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, chosen))


def write_cnf(path: Path, variables: int, clauses: list[list[int]]) -> None:
    path.write_text(f"p cnf {variables} {len(clauses)}\n" + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))


def pigeonhole(holes: int) -> tuple[int, list[list[int]]]:
    var = lambda pigeon, hole: pigeon * holes + hole + 1
    clauses = [[var(p, h) for h in range(holes)] for p in range(holes + 1)]
    clauses += [[-var(a, h), -var(b, h)] for h in range(holes) for a in range(holes + 1) for b in range(a + 1, holes + 1)]
    return (holes + 1) * holes, clauses


# Formula de Tseitin sobre un grafo 3-regular aleatorio con carga impar: siempre UNSAT, y dificil
# para la resolucion. Lineal de resolver por eliminacion gaussiana.
def tseitin(nodes: int, seed: int) -> tuple[int, list[list[int]]]:
    rng = random.Random(seed)
    while True:
        stubs = [node for node in range(nodes) for _ in range(3)]
        rng.shuffle(stubs)
        edges = list(zip(stubs[::2], stubs[1::2]))
        if all(a != b for a, b in edges) and len({tuple(sorted(edge)) for edge in edges}) == len(edges):
            break
    incident: list[list[int]] = [[] for _ in range(nodes)]
    for index, (a, b) in enumerate(edges):
        incident[a].append(index + 1)
        incident[b].append(index + 1)
    clauses = []
    for node, (x, y, z) in enumerate(incident):
        charge = 1 if node == 0 else 0
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    # Prohibe cada asignacion con paridad distinta de la carga.
                    parity = (sx < 0) ^ (sy < 0) ^ (sz < 0)
                    if parity == bool(charge):
                        clauses.append([sx * x, sy * y, sz * z])
    return len(edges), clauses


def random_3sat(variables: int, seed: int) -> tuple[int, list[list[int]]]:
    rng = random.Random(seed)
    clauses = []
    for _ in range(round(variables * 4.26)):
        chosen = rng.sample(range(1, variables + 1), 3)
        clauses.append([literal if rng.random() < 0.5 else -literal for literal in chosen])
    return variables, clauses


def command_generate(_: argparse.Namespace) -> None:
    folder = BENCH / "familias"
    folder.mkdir(parents=True, exist_ok=True)
    for s, t, n in [(3, 5, 13), (3, 5, 14), (4, 4, 17), (4, 4, 18), (3, 6, 17), (3, 6, 18), (4, 5, 24)]:
        subprocess.run([str(MOTOR), "gen", "ramsey", str(n), str(folder / f"ramsey_{s}_{t}_k{n}.cnf"), "--cliques", str(s), str(t)],
                       check=True, capture_output=True)
    for holes in (9, 10, 11, 12):
        write_cnf(folder / f"palomar_{holes}.cnf", *pigeonhole(holes))
    for nodes, seed in [(30, 1), (40, 2), (50, 3), (60, 4), (80, 5)]:
        write_cnf(folder / f"tseitin_{nodes}.cnf", *tseitin(nodes, seed))
    for variables, seeds in [(200, (1, 2, 3)), (250, (1, 2, 3)), (300, (1, 2, 3))]:
        for seed in seeds:
            write_cnf(folder / f"aleatorio_{variables}_{seed}.cnf", *random_3sat(variables, seed))


def read_cnf(path: Path) -> list[list[int]]:
    clauses, current = [], []
    with path.open() as source:
        for line in source:
            if line[:1] in "cp%" or not line.strip():
                if line[:1] == "%":
                    break
                continue
            for token in line.split():
                value = int(token)
                if value:
                    current.append(value)
                else:
                    clauses.append(current)
                    current = []
    return clauses


def model_is_valid(path: Path, stdout: str) -> bool:
    values = set()
    for line in stdout.splitlines():
        if line.startswith("v "):
            values.update(int(token) for token in line[2:].split())
    return all(any(literal in values for literal in clause) for clause in read_cnf(path))


def run_one(name: str, command: list[str], instance: Path, limit: int) -> dict:
    arguments = [part.replace("{cnf}", str(instance)) for part in command]
    if "{cnf}" not in " ".join(command):
        arguments.append(str(instance))
    start = time.perf_counter()
    try:
        done = subprocess.run(arguments, capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired:
        return {"instance": instance.name, "solver": name, "seconds": limit, "verdict": "TIMEOUT", "limit": limit}
    seconds = time.perf_counter() - start
    verdict = {10: "SAT", 20: "UNSAT"}.get(done.returncode, f"ERROR{done.returncode}")
    record = {"instance": instance.name, "solver": name, "seconds": round(seconds, 3), "verdict": verdict, "limit": limit}
    if verdict == "SAT":
        record["model_ok"] = model_is_valid(instance, done.stdout)
    return record


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def parallel(tasks, jobs: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    finished = 0
    with futures.ThreadPoolExecutor(jobs) as pool, output.open("a") as sink:
        for future in futures.as_completed([pool.submit(*task) for task in tasks]):
            record = future.result()
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            finished += 1
            print(f"[{finished}/{len(tasks)}] {record['solver']} {record['instance'][:40]} {record['verdict']} {record['seconds']}", flush=True)


def command_run(options: argparse.Namespace) -> None:
    output = OUT / f"{options.set}.jsonl"
    done = {(record["instance"], record["solver"]) for record in load(output)}
    instances = sorted((BENCH / options.set).glob("*.cnf"))
    tasks = []
    for instance in instances:
        for name, command in solvers().items():
            if (instance.name, name) not in done and (not options.solver or name in options.solver):
                # Un hilo con cuatro a la vez; el enjambre, con sus cuatro, va aparte.
                tasks.append((run_one, name, command, instance, options.limit))
    single = [task for task in tasks if task[1] != "motor-4h"]
    swarm = [task for task in tasks if task[1] == "motor-4h"]
    parallel(single, options.jobs, output)
    parallel(swarm, max(1, options.jobs // 4), output)


def certify_motor(instance: Path, limit: int) -> dict:
    return run_one("motor+veripb", [str(MOTOR), "solve", "{cnf}"], instance, limit)


def certify_cadical(instance: Path, limit: int) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        proof = Path(folder) / "p.drat"
        start = time.perf_counter()
        try:
            solved = subprocess.run([str(CADICAL), "-q", "--no-binary", str(instance), str(proof)], capture_output=True, timeout=limit)
            left = limit - (time.perf_counter() - start)
            checked = subprocess.run([str(DRAT_TRIM), str(instance), str(proof), "-t", str(int(max(left, 1)))], capture_output=True, text=True, timeout=max(left, 1))
        except subprocess.TimeoutExpired:
            return {"instance": instance.name, "solver": "cadical+drat-trim", "seconds": limit, "verdict": "TIMEOUT", "limit": limit}
        seconds = round(time.perf_counter() - start, 3)
    verified = solved.returncode == 20 and "s VERIFIED" in checked.stdout
    return {"instance": instance.name, "solver": "cadical+drat-trim", "seconds": seconds, "verdict": "UNSAT" if verified else f"ERROR{solved.returncode}", "limit": limit}


def command_certify(options: argparse.Namespace) -> None:
    output = OUT / "certificado.jsonl"
    done = {(record["instance"], record["solver"]) for record in load(output)}
    tasks = []
    for set_name in ("sc2023", "familias"):
        records = load(OUT / f"{set_name}.jsonl")
        solved = sorted({r["instance"] for r in records if r["verdict"] == "UNSAT" and r["solver"] == "motor"})
        for name in solved:
            instance = BENCH / set_name / name
            if (name, "motor+veripb") not in done:
                tasks.append((certify_motor, instance, options.limit))
            if (name, "cadical+drat-trim") not in done:
                tasks.append((certify_cadical, instance, options.limit))
    parallel(tasks, options.jobs, output)


def table(records: list[dict], names: list[str], with_rows: bool) -> list[str]:
    by_instance: dict[str, dict[str, dict]] = {}
    for record in records:
        by_instance.setdefault(record["instance"], {})[record["solver"]] = record
    lines = ["| Solver | Resueltas | SAT | UNSAT | Solo este | PAR-2 (s) |", "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        mine = [row[name] for row in by_instance.values() if name in row]
        solved = [r for r in mine if r["verdict"] in ("SAT", "UNSAT")]
        alone = sum(1 for row in by_instance.values() if name in row and row[name]["verdict"] in ("SAT", "UNSAT")
                    and not any(other.split("-")[0] != name.split("-")[0] and r["verdict"] in ("SAT", "UNSAT") for other, r in row.items()))
        par = sum(r["seconds"] if r in solved else PAR_FACTOR * r["limit"] for r in mine)
        lines.append(f"| {name} | {len(solved)}/{len(mine)} | {sum(r['verdict'] == 'SAT' for r in mine)} | {sum(r['verdict'] == 'UNSAT' for r in mine)} | {alone} | {par:.0f} |")
    if with_rows:
        lines += ["", "| Instancia | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for instance in sorted(by_instance):
            cells = []
            for name in names:
                record = by_instance[instance].get(name)
                if record is None:
                    cells.append("-")
                elif record["verdict"] in ("SAT", "UNSAT"):
                    cells.append(f"{record['verdict']} {record['seconds']:.1f}")
                else:
                    cells.append(record["verdict"].lower())
            lines.append(f"| {instance.removesuffix('.cnf')[:24]} | " + " | ".join(cells) + " |")
    return lines


def problems(records: list[dict]) -> list[str]:
    found = []
    by_instance: dict[str, set[str]] = {}
    for record in records:
        by_instance.setdefault(record["instance"], set()).add(record["verdict"])
        if record.get("model_ok") is False:
            found.append(f"Modelo falso: {record['solver']} en {record['instance']}")
        if record["verdict"].startswith("ERROR"):
            found.append(f"Fallo de ejecucion: {record['solver']} en {record['instance']}: {record['verdict']}")
    for instance, verdicts in by_instance.items():
        if {"SAT", "UNSAT"} <= verdicts:
            found.append(f"Veredictos opuestos en {instance}")
    return found


def command_report(_: argparse.Namespace) -> None:
    names = ["kissat", "cadical", "motor", "motor-4h"]
    lines = ["# Comparacion con CaDiCaL y Kissat", "", "Generado por `scripts/comparar_solvers.py report`.", ""]
    lines += [
        "Como leerlo:", "",
        "* `motor` lleva `--no-verify` en las dos primeras tablas; el coste de certificar esta en la ultima.",
        "* Muestra: 60 de las 279 instancias de hasta 1 MiB comprimido, de las 400 de la pista. Quedan fuera las mayores.",
        "* Limite de 120 s en la competicion, donde el real es de 5000 s: favorece a quien responde pronto.",
        "* Kissat rel-4.0.4 no admite Windows: se compilo con un parche minimo y `--quiet`, sin ajustar nada mas.",
        "* Cuatro trabajos a la vez en 16 hilos con nucleos mixtos: los tiempos pequenos son orientativos.",
        "* \"Solo este\" cuenta `motor` y `motor-4h` como un solo solver.",
        "* Kissat no se certifico: la comparacion con prueba es solo contra CaDiCaL.", "",
    ]
    for set_name, title in (("sc2023", "SAT Competition 2023, pista main"), ("familias", "Familias generadas")):
        records = load(OUT / f"{set_name}.jsonl")
        if not records:
            continue
        limit = records[0]["limit"]
        lines += [f"## {title} (limite de {limit} s por instancia)", ""] + table(records, names, True) + [""]
    certified = load(OUT / "certificado.jsonl")
    if certified:
        lines += ["## Con certificado", "", "Tiempo total de resolver y comprobar la prueba, en los UNSAT que `motor` resolvio.", ""]
        lines += table(certified, ["motor+veripb", "cadical+drat-trim"], True) + [""]
    every = [record for set_name in ("sc2023", "familias", "certificado") for record in load(OUT / f"{set_name}.jsonl")]
    found = problems(every)
    lines += ["## Comprobaciones", "", f"{len(every)} ejecuciones. Cada modelo se evaluo contra la CNF, sin creer al solver.", ""]
    lines += [f"* {item}" for item in found] or ["Ningun modelo falso, ningun veredicto opuesto, ningun fallo de ejecucion."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    for item in found:
        print(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("tools", command_tools), ("fetch", command_fetch), ("generate", command_generate), ("report", command_report)):
        commands.add_parser(name).set_defaults(handler=handler)
    run = commands.add_parser("run")
    run.add_argument("set", choices=["sc2023", "familias"])
    run.add_argument("--limit", type=int, default=120)
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--solver", action="append")
    run.set_defaults(handler=command_run)
    certify = commands.add_parser("certify")
    certify.add_argument("--limit", type=int, default=600)
    certify.add_argument("--jobs", type=int, default=4)
    certify.set_defaults(handler=command_certify)
    options = parser.parse_args()
    options.handler(options)


if __name__ == "__main__":
    main()
