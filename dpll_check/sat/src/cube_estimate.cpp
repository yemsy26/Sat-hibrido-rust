// Estimador de Knuth para cube-and-conquer con orden estatico de variables.
//
// Arbol: se recorren las variables de <orden> en secuencia; una variable ya
// implicada por propagacion unitaria (UP) no ramifica; un valor cuya UP da
// conflicto se poda (como hace un cubeador lookahead con literales
// fallidos). Las hojas son los nodos con k variables fijadas por el cubo
// (forzadas o decididas), y cada hoja se resuelve con CDCL en una copia
// fresca de CaDiCaL con limite de tiempo.
//
// La UP es propia y sin estado: CaDiCaL::propagate() aprende unidades de los
// conflictos, lo que haria depender el arbol del orden de muestreo.
//
// Modo muestreo (samples > 0): camino aleatorio raiz-hoja; peso = 2^(nodos
// con dos hijos vivos). E[peso] = #hojas y E[peso * t_hoja] = coste total
// (ambos insesgados). Modo exacto (samples = -1): recorre todas las hojas.
// Modo poblacion (samples = -P, P >= 2): ver population(); imprime ademas
// "L <nivel> <log2 nodos estimados> <nodos expandidos>" por nivel.
//
// Uso: cube_estimate <cnf> <orden.txt> <k> <samples|-1|-P> <seg_hoja> <semilla>
//        [cubos_salida]   (si se da, escribe el cubo de cada hoja: literales y 0)
// Salida por hoja: "<log2_peso> <fijadas> <SAT|UNSAT|TIMEOUT|PRUNED> <s> <conflictos>
// <ramas>", con <ramas> = profundidades (separadas por comas, '-' si ninguna)
// de los nodos del camino con dos hijos vivos.
#include "cadical.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Deadline : CaDiCaL::Terminator {
  Clock::time_point end;
  bool terminate() override { return Clock::now() >= end; }
};

// Propagacion unitaria con dos literales vigilados; deshace todo al salir.
struct UnitProp {
  std::vector<std::vector<int>> cls;
  std::vector<std::vector<int>> watch;  // por literal codificado
  std::vector<signed char> val;         // por variable: 0, 1, -1
  std::vector<int> units, trail;
  bool empty = false;

  static size_t code(int lit) { return 2 * (size_t)std::abs(lit) + (lit < 0); }
  int value(int lit) const {
    int v = val[std::abs(lit)];
    return lit > 0 ? v : -v;
  }
  bool assign(int lit) {
    int v = value(lit);
    if (v) return v > 0;
    val[std::abs(lit)] = lit > 0 ? 1 : -1;
    trail.push_back(lit);
    return true;
  }
  void init(int vars) {
    val.assign(vars + 1, 0);
    watch.assign(2 * (size_t)vars + 2, {});
    for (size_t i = 0; i < cls.size(); i++) {
      auto &c = cls[i];
      if (c.empty()) empty = true;
      else if (c.size() == 1) units.push_back(c[0]);
      else watch[code(c[0])].push_back(i), watch[code(c[1])].push_back(i);
    }
  }
  // false si UP(unidades + supuestos) da conflicto; si no, implied = traza.
  bool run(const std::vector<int> &assume, std::vector<int> &implied) {
    bool ok = !empty;
    for (int l : units) ok = ok && assign(l);
    for (int l : assume) ok = ok && assign(l);
    for (size_t head = 0; ok && head < trail.size(); head++) {
      int f = -trail[head];  // literal que acaba de hacerse falso
      auto &ws = watch[code(f)];
      for (size_t i = 0; i < ws.size();) {
        auto &c = cls[ws[i]];
        if (c[0] == f) std::swap(c[0], c[1]);
        if (value(c[0]) > 0) { i++; continue; }
        size_t k = 2;
        while (k < c.size() && value(c[k]) < 0) k++;
        if (k < c.size()) {
          std::swap(c[1], c[k]);
          watch[code(c[1])].push_back(ws[i]);
          ws[i] = ws.back();
          ws.pop_back();
          continue;
        }
        if (value(c[0]) < 0) { ok = false; break; }
        assign(c[0]);
        i++;
      }
    }
    implied.clear();
    if (ok) implied = trail;
    for (int l : trail) val[std::abs(l)] = 0;
    trail.clear();
    return ok;
  }
};

static CaDiCaL::Solver base;
static UnitProp up;
static std::vector<int> order;
static int K;
static double limit;
static FILE *cube_out;
static std::string branches;  // perfil del camino actual

static void leaf(const std::vector<int> &path, double log2w, int fixed) {
  if (cube_out) {
    for (int l : path) std::fprintf(cube_out, "%d ", l);
    std::fprintf(cube_out, "0\n");
    std::fflush(cube_out);
  }
  CaDiCaL::Solver s;
  base.copy(s);
  s.set("quiet", 1);
  Deadline d;
  auto t0 = Clock::now();
  d.end = t0 + std::chrono::duration_cast<Clock::duration>(
                   std::chrono::duration<double>(limit));
  s.connect_terminator(&d);
  for (int l : path) s.assume(l);
  int res = s.solve();
  double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("%.3f %d %s %.4f %lld %s\n", log2w, fixed,
              res == 10 ? "SAT" : res == 20 ? "UNSAT" : "TIMEOUT", secs,
              (long long)s.get_statistic_value("conflicts"),
              branches.empty() ? "-" : branches.c_str());
  std::fflush(stdout);
}

// Hijos vivos del nodo `path`; avanza `pos` sobre variables ya implicadas.
static std::vector<int> children(const std::vector<int> &path, size_t &pos) {
  std::vector<int> imp, tmp, alive;
  up.run(path, imp);
  std::vector<char> set(up.val.size(), 0);
  for (int l : imp) set[std::abs(l)] = 1;
  while (pos < order.size() && set[order[pos]]) pos++;
  if (pos == order.size()) return alive;
  std::vector<int> p = path;
  p.push_back(0);
  for (int l : {order[pos], -order[pos]}) {
    p.back() = l;
    if (up.run(p, tmp)) alive.push_back(l);
  }
  return alive;
}

static void exact(std::vector<int> &path, size_t pos) {
  if ((int)path.size() == K) return leaf(path, 0, K);
  std::vector<int> ch = children(path, pos);
  if (ch.empty()) {
    if (pos == order.size()) leaf(path, 0, path.size());
    return;  // si no, ambos valores fallan: nodo refutado por UP
  }
  for (int l : ch) {
    path.push_back(l);
    exact(path, pos + 1);
    path.pop_back();
  }
}

// Poblacion (remuestreo sistematico, "go with the winners"): se mantienen
// hasta P nodos por nivel con pesos; la suma de pesos estima sin sesgo el
// numero de nodos vivos del nivel aunque casi todos los caminos mueran.
struct Node {
  std::vector<int> path;
  size_t pos;
  double w;
};

static void population(long P, std::mt19937_64 &rng) {
  std::vector<Node> pop{{{}, 0, 1.0}}, done;
  for (int j = 0; j < K && !pop.empty(); j++) {
    std::vector<Node> next;
    for (Node &nd : pop) {
      std::vector<int> ch = children(nd.path, nd.pos);
      if (ch.empty() && nd.pos == order.size()) done.push_back(nd);
      for (int l : ch) {
        next.push_back({nd.path, nd.pos + 1, nd.w});
        next.back().path.push_back(l);
      }
    }
    double W = 0;
    for (Node &nd : next) W += nd.w;
    std::printf("L %d %.4f %zu\n", j + 1, W > 0 ? std::log2(W) : -1.0,
                next.size());
    std::fflush(stdout);
    if ((long)next.size() > P) {
      std::vector<Node> keep;
      double step = W / P, u = std::uniform_real_distribution<>(0, step)(rng);
      double acc = 0;
      for (Node &nd : next) {
        acc += nd.w;
        while (u < acc && (long)keep.size() < P) {
          keep.push_back({nd.path, nd.pos, step});
          u += step;
        }
      }
      next.swap(keep);
    }
    pop.swap(next);
  }
  for (Node &nd : done) pop.push_back(nd);
  for (Node &nd : pop) {
    branches.clear();
    leaf(nd.path, std::log2(nd.w), nd.path.size());
  }
}

int main(int argc, char **argv) {
  if (argc < 7) {
    std::fprintf(stderr, "uso: %s <cnf> <orden> <k> <samples> <seg> <semilla>\n",
                 argv[0]);
    return 2;
  }
  K = std::atoi(argv[3]);
  long samples = std::atol(argv[4]);
  limit = std::atof(argv[5]);
  std::mt19937_64 rng(std::strtoull(argv[6], 0, 10));
  if (argc > 7 && !(cube_out = std::fopen(argv[7], "w"))) return 2;

  FILE *f = std::fopen(argv[1], "r");
  if (!f) return 2;
  int vars = 0, ncls = 0, c;
  while ((c = std::fgetc(f)) == 'c')
    while ((c = std::fgetc(f)) != '\n' && c != EOF) {}
  if (c != 'p' || std::fscanf(f, " cnf %d %d", &vars, &ncls) != 2) return 2;
  up.cls.reserve(ncls);
  std::vector<int> cl;
  for (int lit; std::fscanf(f, "%d", &lit) == 1;) {
    base.add(lit);
    if (lit) cl.push_back(lit);
    else up.cls.push_back(cl), cl.clear();
  }
  std::fclose(f);
  base.set("quiet", 1);
  up.init(vars);

  f = std::fopen(argv[2], "r");
  if (!f) return 2;
  for (int v; std::fscanf(f, "%d", &v) == 1;) order.push_back(v);
  std::fclose(f);

  std::vector<int> path;
  if (samples == -1) {
    exact(path, 0);
    return 0;
  }
  if (samples < -1) {
    population(-samples, rng);
    return 0;
  }
  for (long i = 0; i < samples; i++) {
    path.clear();
    branches.clear();
    size_t pos = 0;
    double log2w = 0;
    bool pruned = false;
    while ((int)path.size() < K) {
      std::vector<int> ch = children(path, pos);
      if (ch.empty()) {
        pruned = pos < order.size();
        break;
      }
      if (ch.size() == 2) {
        log2w += 1;
        if (!branches.empty()) branches += ',';
        branches += std::to_string(path.size());
      }
      path.push_back(ch[ch.size() == 2 ? rng() % 2 : 0]);
      pos++;
    }
    if (pruned) {
      std::printf("%.3f %zu PRUNED 0 0 %s\n", log2w, path.size(),
                  branches.empty() ? "-" : branches.c_str());
      std::fflush(stdout);
    } else {
      leaf(path, log2w, path.size());
    }
  }
  return 0;
}
