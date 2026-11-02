# Extensiones de los 656 grafos Ramsey(5,5,42) conocidos a K43

Datos: `https://users.cecs.anu.edu.au/~bdm/data/r55_42some.g6` (SHA-256 `067902e853d87b49bcef0d1d4c0e3bbadd238ee18bc65341b079a3ca4780eccb`), 328 grafos y sus complementos.
Motor: `gpu_ramsey43 --base <g>.sat --lookahead --explicit-trail --epochs 0`; reproducir con `python -I scripts/r55_42.py all`.

| Comprobación | Bases | Resultado |
|---|---|---|
| `check_reference` sin K5 en la parte fija, estrella de 0 libre | 656 | 656 correctas |
| GPU `s UNSATISFIABLE` | 656 | 656/656 |
| CPU DPLL independiente UNSAT | 656 | 656/656 |
| CPU control positivo (sin vértice 42 extiende) | 656 | 656/656 |
| Desacuerdos GPU/CPU | 0 | ninguno |

| Métrica GPU (total / mediana / máx.) | Valor |
|---|---|
| elapsed_ms del motor | 1515784 / 1985.9 / 8214.5 |
| transiciones | 797185388 / 1150612.0 / 2253506.0 |
| decisiones | 225750684 / 335690.0 / 609104.0 |
| tiempo de pared por proceso (s) | 4523 / 5.2 / 39.7 |
| nodos DPLL en CPU | 26352 / 39.0 / 51.0 |

Excepciones: ninguna.

Detalle por grafo: `gpu.csv`, `cpu.csv`, `audit.csv`.
