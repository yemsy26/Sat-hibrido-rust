// Enumera los modelos de una CNF proyectados sobre las variables 1..E
// (las aristas), bloqueando cada proyeccion encontrada.
// Uso: enum_models <cnf> <E> [max_modelos]
// Salida: una linea por modelo con E caracteres '0'/'1' (1 = rojo).
#include "cadical.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "uso: %s <cnf> <E> [max]\n", argv[0]);
    return 2;
  }
  int E = std::atoi(argv[2]);
  long max = argc > 3 ? std::atol(argv[3]) : -1;
  CaDiCaL::Solver s;
  s.set("quiet", 1);
  int vars = 0;
  if (const char *err = s.read_dimacs(argv[1], vars)) {
    std::fprintf(stderr, "error: %s\n", err);
    return 2;
  }
  long count = 0;
  std::string line(E, '0');
  while ((max < 0 || count < max) && s.solve() == 10) {
    for (int v = 1; v <= E; v++) line[v - 1] = s.val(v) > 0 ? '1' : '0';
    for (int v = 1; v <= E; v++) s.add(line[v - 1] == '1' ? -v : v);
    s.add(0);
    std::puts(line.c_str());
    count++;
  }
  std::fprintf(stderr, "modelos: %ld%s\n", count,
               (max >= 0 && count == max) ? " (tope alcanzado)" : "");
  return 0;
}
