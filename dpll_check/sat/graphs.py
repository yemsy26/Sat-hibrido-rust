"""Utilidades de grafos pequenos: isomorfismo por backtracking y clasificacion.

Independiente de la CNF: solo se usa para auditar que la ruptura de
simetria no pierde clases de isomorfismo.
"""
import itertools


def from_bits(n, bits):
    """bits[k] = '1' si la k-esima arista (orden fila a fila) es roja."""
    adj = [[False] * n for _ in range(n)]
    for k, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        adj[i][j] = adj[j][i] = bits[k] == "1"
    return adj


def complement(adj):
    n = len(adj)
    return [[i != j and not adj[i][j] for j in range(n)] for i in range(n)]


def vertex_inv(adj):
    n = len(adj)
    deg = [sum(r) for r in adj]
    inv = []
    for v in range(n):
        nb = [u for u in range(n) if adj[v][u]]
        tri = sum(adj[a][b] for a, b in itertools.combinations(nb, 2))
        inv.append((deg[v], tri, tuple(sorted(deg[u] for u in nb))))
    return inv


def graph_key(adj):
    return tuple(sorted(vertex_inv(adj)))


def isomorphic(A, B):
    n = len(A)
    ia, ib = vertex_inv(A), vertex_inv(B)
    if sorted(ia) != sorted(ib):
        return False
    order = sorted(range(n), key=lambda v: (ia.count(ia[v]), -ia[v][0]))
    img = [-1] * n
    used = [False] * n

    def go(k):
        if k == n:
            return True
        v = order[k]
        for w in range(n):
            if used[w] or ib[w] != ia[v]:
                continue
            if all(A[v][order[m]] == B[w][img[order[m]]] for m in range(k)):
                img[v], used[w] = w, True
                if go(k + 1):
                    return True
                used[w] = False
        img[v] = -1
        return False

    return go(0)


class Classifier:
    """Agrupa grafos en clases de isomorfismo (opcionalmente modulo complemento)."""

    def __init__(self, up_to_complement=False):
        self.cc = up_to_complement
        self.buckets = {}
        self.count = 0

    def add(self, adj):
        forms = [adj, complement(adj)] if self.cc else [adj]
        key = min(graph_key(g) for g in forms)
        reps = self.buckets.setdefault(key, [])
        for r in reps:
            if any(isomorphic(g, r) for g in forms):
                return False
        reps.append(adj)
        self.count += 1
        return True
