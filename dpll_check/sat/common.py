"""Rutas de herramientas y particion en clases de raiz compartidas por los scripts."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "tools")
EXE = ".exe" if os.name == "nt" else ""
CADICAL = os.path.join(TOOLS, "cadical", "build", "cadical" + EXE)
LRAT_CHECK = os.path.join(TOOLS, "drat-trim", "lrat-check" + EXE)
DRAT_TRIM = os.path.join(TOOLS, "drat-trim", "drat-trim" + EXE)
ENUM = os.path.join(HERE, "bin", "enum_models" + EXE)
WORK = os.path.join(HERE, "work")
RESULTS = os.path.join(HERE, "results")


def root_classes(s, t, n, deg_range=None):
    """Clases (d_raiz, ventana) que cubren TODA coloracion de K_n.

    Sea m = min_v min(g(v), n-1-g(v)) con g = grado rojo. Se elige como raiz
    un v que lo alcanza; entonces todo grado esta en [m, n-1-m] y la raiz
    tiene grado rojo m o n-1-m. Si s == t, intercambiar colores es una
    simetria y basta grado rojo m. Se omite m = (n-1)/2 impar con n impar
    (grafo regular de grado impar en n impar: lema del apreton de manos).
    deg_range=(a, b): cota de grado rojo valida para todo vertice, derivada
    de numeros de Ramsey menores (es un LEMA externo a la prueba).
    """
    out = []
    for m in range((n - 1) // 2 + 1):
        lo, hi = m, n - 1 - m
        if lo == hi and n % 2 == 1 and lo % 2 == 1:
            continue
        roots = [lo] if (s == t or lo == hi) else [lo, hi]
        for d in roots:
            w = (lo, hi)
            if deg_range:
                a, b = deg_range
                if not a <= d <= b:
                    continue  # la raiz violaria el lema: clase vacia
                w = (max(lo, a), min(hi, b))
            out.append((d, w))
    return out


def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
