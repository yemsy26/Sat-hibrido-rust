"""Vigila DIMACS, conserva el modelo y audita ambos colores de Ramsey."""

from __future__ import annotations

import argparse
import codecs
import itertools
import math
import os
from pathlib import Path
import sys
import time


class AuditError(ValueError):
    """Un modelo incompleto o una coloracion que no supera la auditoria."""


def text_encoding(prefix: bytes) -> str:
    if prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if prefix.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if len(prefix) >= 4 and prefix[1] == prefix[3] == 0:
        return "utf-16-le"
    if len(prefix) >= 4 and prefix[0] == prefix[2] == 0:
        return "utf-16-be"
    return "utf-8"


class LogTail:
    """Lee desde el inicio; retiene lineas parciales y detecta rotacion/truncado."""

    def __init__(self, path: Path):
        self.path = path
        self.identity = None
        self.position = 0
        self.checkpoint = b""
        self.decoder = None
        self.prefix = b""
        self.pending = ""

    def close(self) -> None:
        self.identity = self.decoder = None
        self.position = 0
        self.checkpoint = b""
        self.prefix = b""
        self.pending = ""

    def poll(self) -> tuple[list[str], bool]:
        try:
            # Reabrir cada lectura permite rotar el archivo tambien en Windows.
            stream = self.path.open("rb")
        except FileNotFoundError:
            reset = self.identity is not None
            self.close()
            return [], reset
        with stream:
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino)
            reset = (self.identity is None or identity != self.identity
                     or info.st_size < self.position)
            if not reset and self.checkpoint:
                stream.seek(self.position - len(self.checkpoint))
                reset = stream.read(len(self.checkpoint)) != self.checkpoint
            if reset:
                self.close()
                self.identity = identity
            stream.seek(self.position)
            data = stream.read(65536)
            self.position = stream.tell()
            checkpoint_size = min(64, self.position)
            stream.seek(self.position - checkpoint_size)
            self.checkpoint = stream.read(checkpoint_size)
        if not data:
            return [], reset
        if self.decoder is None:
            self.prefix += data
            if len(self.prefix) < 4:
                return [], reset
            self.decoder = codecs.getincrementaldecoder(
                text_encoding(self.prefix)
            )(errors="replace")
            data, self.prefix = self.prefix, b""
        self.pending += self.decoder.decode(data)
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        return [line.rstrip("\r") for line in lines], reset


class ModelCollector:
    def __init__(self):
        self.active = False
        self.done = False
        self.values: dict[int, bool] = {}

    def feed(self, line: str) -> str | None:
        tokens = line.split()
        if not tokens or tokens[0].lower() == "c":
            return None
        status = " ".join(tokens).upper()
        if status in {"S SATISFIABLE", "S SAT", "SATISFIABLE", "SAT"}:
            self.active = True
            return "sat"
        if status in {"S UNSATISFIABLE", "S UNSAT", "UNSATISFIABLE", "UNSAT"}:
            return "unsat"
        if status in {"S UNKNOWN", "UNKNOWN"}:
            return "unknown"
        if not self.active or tokens[0].lower() != "v":
            return None
        if self.done:
            raise AuditError("Hay literales despues del terminador del modelo.")
        for index, token in enumerate(tokens[1:], 1):
            try:
                literal = int(token)
            except ValueError as error:
                raise AuditError(f"Literal DIMACS invalido: {token!r}") from error
            if literal == 0:
                if index != len(tokens) - 1:
                    raise AuditError("El terminador 0 debe cerrar la linea v.")
                self.done = True
                return "model"
            variable, value = abs(literal), literal > 0
            if variable in self.values and self.values[variable] != value:
                raise AuditError(f"Asignaciones contradictorias de x{variable}.")
            self.values[variable] = value
        return None


def read_model_file(path: Path) -> ModelCollector | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    if len(data) < 4:
        return None
    candidate = ModelCollector()
    candidate.feed("s SATISFIABLE")
    # Una escritura concurrente puede terminar en media unidad UTF-16.
    decoder = codecs.getincrementaldecoder(text_encoding(data))(errors="strict")
    text = decoder.decode(data, final=False)
    # No consumir la ultima linea si la escritura todavia esta incompleta.
    for line in text.splitlines(keepends=True):
        if not line.endswith(("\n", "\r")):
            break
        candidate.feed(line)
    return candidate if candidate.done else None


def file_signature(path: Path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def audit_model(values: dict[int, bool], vertices: int):
    import networkx as nx

    edge_count = vertices * (vertices - 1) // 2
    missing = [variable for variable in range(1, edge_count + 1)
               if variable not in values]
    if missing:
        raise AuditError(
            f"Faltan {len(missing)} aristas; primeras variables: {missing[:8]}"
        )
    graph = nx.Graph()
    graph.add_nodes_from(range(1, vertices + 1))
    # Coincide con gen_ramsey: (1,2), (1,3), ..., (2,3), ... .
    for variable, edge in enumerate(itertools.combinations(graph.nodes, 2), 1):
        if values[variable]:
            graph.add_edge(*edge)
    complement = nx.complement(graph)
    maxima = []
    for color, colored_graph in ((1, graph), (0, complement)):
        maximum = 0
        for clique in nx.find_cliques(colored_graph):
            if len(clique) >= 5:
                raise AuditError(f"K_5 del color {color}: {sorted(clique)[:5]}")
            maximum = max(maximum, len(clique))
        maxima.append(maximum)
    return graph, complement, tuple(maxima)


def render_graph(graph, complement, maxima, output: Path, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 12))
    positions = nx.circular_layout(graph)
    nx.draw_networkx_edges(complement, positions, ax=axes,
                           edge_color="steelblue", alpha=0.22, width=0.6)
    nx.draw_networkx_edges(graph, positions, ax=axes,
                           edge_color="firebrick", alpha=0.65, width=0.7)
    nx.draw_networkx_nodes(graph, positions, ax=axes, node_size=110,
                           node_color="white", edgecolors="black")
    nx.draw_networkx_labels(graph, positions, ax=axes, font_size=7)
    axes.legend(handles=[
        Line2D([0], [0], color="firebrick", label=f"Color 1: clique maximo {maxima[0]}"),
        Line2D([0], [0], color="steelblue", label=f"Color 0: clique maximo {maxima[1]}"),
    ], loc="upper right")
    axes.set_title(f"K_{graph.number_of_nodes()}: coloracion auditada sin K_5 monocromatico")
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    print(f"Topologia guardada: {output}", flush=True)
    if show:
        plt.show()
    plt.close(figure)


def monitor(options) -> int:
    collector = ModelCollector()
    sat_since = None
    model_baseline = file_signature(options.model_path)
    tail = LogTail(options.log)
    print(f"Centinela vigilando: {options.log}", flush=True)
    print(f"Modelo DIMACS del motor: {options.model_path}", flush=True)
    print("Ctrl+C detiene solo el centinela.", flush=True)
    try:
        while True:
            previous_identity = tail.identity
            lines, reset = tail.poll()
            if reset:
                collector = ModelCollector()
                sat_since = None
                if previous_identity is not None:
                    model_baseline = file_signature(options.model_path)
            for line in lines:
                event = collector.feed(line)
                if event == "sat" and sat_since is None:
                    sat_since = time.monotonic()
                    print("SAT detectado; esperando el modelo completo...", flush=True)
                elif event == "unsat":
                    print("UNSAT reportado por el motor. No hay modelo ni topologia.")
                    return 0
                elif event == "unknown":
                    print("UNKNOWN reportado por el motor. No hay modelo certificado.")
                    return 3
            if sat_since is not None and not collector.done:
                signature = file_signature(options.model_path)
                candidate = None
                if signature is not None and (
                    options.allow_existing_model or signature != model_baseline
                ):
                    candidate = read_model_file(options.model_path)
                    if signature != file_signature(options.model_path):
                        candidate = None
                if candidate is not None:
                    collector = candidate
                    print(f"Modelo candidato leido de: {options.model_path}", flush=True)
                elif time.monotonic() - sat_since >= options.model_wait:
                    raise AuditError(
                        "SAT sin modelo completo. El motor necesita --model modelo.sat "
                        "si no imprime lineas v en el log. Para auditar un archivo "
                        "ya existente, usa --allow-existing-model explicitamente."
                    )
            if collector.done:
                saved = options.output.with_suffix(".modelo.sat")
                saved.parent.mkdir(parents=True, exist_ok=True)
                literals = [str(variable if value else -variable)
                            for variable, value in sorted(collector.values.items())]
                saved.write_text("v " + " ".join(literals) + " 0\n", encoding="utf-8")
                print(f"Modelo capturado y conservado: {saved}", flush=True)
                graph, complement, maxima = audit_model(collector.values, options.vertices)
                print(f"AUDITORIA OK: color 1, clique maximo {maxima[0]} < 5; "
                      f"color 0, clique maximo {maxima[1]} < 5.", flush=True)
                render_graph(graph, complement, maxima, options.output, not options.no_show)
                return 0
            if not lines:
                time.sleep(options.poll)
    finally:
        tail.close()


def main(argv=None) -> int:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path,
                        default=directory / "results" / "modelo_k43_neuro.sat")
    parser.add_argument("--model-path", type=Path,
                        default=directory / "results" / "modelo_k43_neuro.sat")
    parser.add_argument("--output", type=Path,
                        default=directory / "results" / "k43_topologia_descubierta.png")
    parser.add_argument("--vertices", type=int, default=43)
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--model-wait", type=float, default=60.0)
    parser.add_argument("--allow-existing-model", action="store_true",
                        help="Auditar un archivo ya existente, sin atribuirlo al proceso actual")
    parser.add_argument("--no-show", action="store_true", help="Guardar PNG sin abrir ventana")
    options = parser.parse_args(argv)
    if (options.vertices < 2 or not math.isfinite(options.poll)
            or not math.isfinite(options.model_wait)
            or options.poll <= 0 or options.model_wait <= 0):
        parser.error("vertices >= 2, poll > 0 y model-wait > 0 son obligatorios")
    try:
        return monitor(options)
    except KeyboardInterrupt:
        print("\nCentinela detenido.")
        return 130
    except (AuditError, OSError, UnicodeError, ImportError) as error:
        print(f"ERROR del centinela: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
