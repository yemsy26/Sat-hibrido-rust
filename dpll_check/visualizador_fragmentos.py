"""Visualiza bajo demanda una instantanea parcial del trail; no certifica SAT."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from centinela_k43 import AuditError, ModelCollector, text_encoding


@dataclass(frozen=True)
class Fragment:
    values: dict[int, bool]
    depth: int
    total_variables: int


def read_fragment(path: Path, total_variables: int = 903) -> Fragment:
    """Lee una sola version completa del archivo publicado por el motor."""
    if total_variables <= 0:
        raise AuditError("El total de variables debe ser positivo.")
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise AuditError(
            f"Todavia no existe {path}. Proporciona una instantanea DIMACS parcial "
            "con --input. El motor CUDA actual no exporta fragmentos del trail."
        ) from error
    text = data.decode(text_encoding(data), errors="strict")
    collector = ModelCollector()
    # Solo activa el lector DIMACS; el fragmento no constituye un modelo SAT.
    collector.feed("s SATISFIABLE")
    metadata = None
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].lower() == "c":
            if len(tokens) >= 2 and tokens[1].lower() == "depth":
                if metadata is not None:
                    raise AuditError("Hay dos cabeceras de profundidad.")
                if len(tokens) != 5 or tokens[3].lower() != "total":
                    raise AuditError("Cabecera esperada: c depth X total Y.")
                try:
                    metadata = int(tokens[2]), int(tokens[4])
                except ValueError as error:
                    raise AuditError("La profundidad y el total deben ser enteros.") from error
            continue
        if tokens[0].lower() != "v":
            raise AuditError("El fragmento admite solamente comentarios c y lineas v.")
        collector.feed(line)
    if not collector.done:
        raise AuditError("Fragmento incompleto: falta el terminador DIMACS 0.")
    depth = len(collector.values)
    if metadata is not None:
        declared_depth, total_variables = metadata
        if declared_depth != depth:
            raise AuditError(
                f"La cabecera declara profundidad {declared_depth}, "
                f"pero hay {depth} variables distintas."
            )
    if total_variables <= 0 or depth > total_variables:
        raise AuditError("Profundidad o total de variables incompatible.")
    # Los IDs DIMACS externos pueden ser dispersos: total cuenta variables,
    # no limita el valor numerico de sus identificadores.
    return Fragment(collector.values, depth, total_variables)


def positive_graph(fragment: Fragment, vertices: int):
    """IDs 1..n(n-1)/2 corresponden a las parejas en orden lexicografico."""
    import networkx as nx

    if vertices < 2:
        raise AuditError("Se necesitan al menos dos vertices.")
    graph = nx.Graph()
    graph.add_nodes_from(range(1, vertices + 1))
    for variable, edge in enumerate(itertools.combinations(graph.nodes, 2), 1):
        if fragment.values.get(variable) is True:
            graph.add_edge(*edge)
    return graph


def describe_fragment(fragment: Fragment, vertices: int) -> str:
    edge_variables = vertices * (vertices - 1) // 2
    positives = sum(value for variable, value in fragment.values.items()
                    if variable <= edge_variables)
    negatives = sum(not value for variable, value in fragment.values.items()
                    if variable <= edge_variables)
    auxiliaries = sum(variable > edge_variables for variable in fragment.values)
    unknown = edge_variables - positives - negatives
    return (f"Profundidad: {fragment.depth}/{fragment.total_variables}; "
            f"aristas positivas: {positives}, negativas: {negatives}, "
            f"sin asignar: {unknown}; auxiliares asignadas: {auxiliaries}.")


def render_fragment(fragment: Fragment, vertices: int, output: Path,
                    show: bool = True) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = positive_graph(fragment, vertices)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 12))
    positions = nx.circular_layout(graph)
    nx.draw_networkx_edges(graph, positions, ax=axes,
                           edge_color="firebrick", alpha=0.6, width=0.7)
    nx.draw_networkx_nodes(graph, positions, ax=axes, node_size=130,
                           node_color="white", edgecolors="black")
    nx.draw_networkx_labels(graph, positions, ax=axes, font_size=7)
    axes.set_title("Near-Miss: Grafo parcial construido "
                   f"(Profundidad: {fragment.depth}/{fragment.total_variables})")
    axes.set_axis_off()
    figure.text(0.5, 0.035,
                "Solo aristas positivas. La profundidad incluye auxiliares.\n"
                "Instantanea parcial de busqueda; no certifica SAT ni ausencia de K5.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output, dpi=200)
    print(f"Grafo parcial guardado: {output}", flush=True)
    if show:
        plt.show()
    plt.close(figure)


def main(argv=None) -> int:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=directory / "results" / "fragmento_profundo.sat")
    parser.add_argument("--output", type=Path,
                        default=directory / "results" / "fragmento_profundo.png")
    parser.add_argument("--vertices", type=int, default=43)
    parser.add_argument("--total-variables", type=int, default=903,
                        help="Total usado si el archivo no incluye c depth X total Y")
    parser.add_argument("--no-show", action="store_true",
                        help="Guardar PNG sin abrir una ventana")
    options = parser.parse_args(argv)
    if options.vertices < 2 or options.total_variables <= 0:
        parser.error("vertices >= 2 y total-variables > 0 son obligatorios")
    try:
        fragment = read_fragment(options.input, options.total_variables)
        print(describe_fragment(fragment, options.vertices), flush=True)
        print("La profundidad cuenta variables auxiliares; no es un certificado SAT.",
              flush=True)
        render_fragment(fragment, options.vertices, options.output,
                        show=not options.no_show)
        return 0
    except KeyboardInterrupt:
        return 130
    except (AuditError, OSError, UnicodeError, ImportError) as error:
        print(f"ERROR del visualizador: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
