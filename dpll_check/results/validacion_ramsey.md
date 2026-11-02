# Validación del DFS sobre números de Ramsey conocidos

Banco que comprueba si el motor DFS de `src/ramsey43_native.cu` resuelve casos de Ramsey
ya demostrados, y si la política neuronal (`neuro_ramsey_policy.onnx`) reduce el árbol.
No toca el lanzador del asedio K43 (`gpu_ramsey43`, `scripts/`, `*.bat`).

## Qué se ejecuta

- **`src/ramsey_ref.rs`**: es la referencia CPU de `ramsey43_search_journal_body` (un universo,
  `prefix_bits=0`), generalizada a K_n sin K_s rojo ni K_t azul. Usa la misma primera arista
  libre lexicográfica, el mismo primer color por hash, el mismo diario de frames y los mismos
  contadores. El oráculo `lookahead` combina el K_s/K_t nuevo con las cotas de grado
  R(s-1,t)-1 (rojo) y R(s,t-1)-1 (azul). Con (5,5,43) da las 24/24 del kernel.
- **Fidelidad con la GPU**: `tests/ramsey_gpu_parity.rs` ejecuta los kernels
  `ramsey43_search_journal` y `ramsey43_search_journal_smart` y la referencia sobre las 4
  estrellas K43, con y sin lookahead y con las semillas 0 y 0x5eed: 16 ejecuciones de 524 288
  transiciones cada una. Transiciones, podas, decisiones, forzadas, checks y conflictos
  coinciden **exactamente**. Se ejecutó en una RTX 4060 Laptop con NVRTC 12.6. Los resultados
  de abajo son, por tanto, los del motor GPU en modo journal.
- **Política neuronal**: la MLP 32→32→32→2 del ONNX se evalúa en CPU, en f32, con las entradas
  redondeadas a binary16 como en `ramsey43_pack_features`. El protocolo es el de
  `ramsey43_search_neuro_smart`: una consulta cada `interval` decisiones libres, sobre las 32
  primeras aristas libres, eligiendo el mínimo coste, revalidado después por el oráculo. Las
  features se normalizan por |E| y n−1 en lugar de 903 y 42. La ruta TensorRT no se ejecutó
  porque `neuro-trt` no compila: `build.rs` fija MSVC 14.44 y la máquina tiene la 14.51.
- **Control `azar/k`**: misma cadencia y los mismos 32 candidatos que `neuro/k`, pero arista y
  color uniformes. Sirve para separar lo que aporta la red de lo que aporta perturbar el orden.
- **Heurísticas**:
  - `lex`: el kernel base, sin lookahead; sólo poda por K nuevo.
  - `lex+LA`: `--lookahead`.
  - `fail-first+LA`: MRV sobre todas las aristas libres. Conflicto si alguna tiene 0 colores
    permitidos; asignación forzada si tiene 1; si no, la de más aristas asignadas en sus
    extremos y luego la de más vecinos comunes del mismo color.
  - `neuro/32+LA`: el intervalo por defecto del lanzador.
  - `neuro/1+LA`: consulta en cada decisión.
- **Raíces**:
  - `vacía`: sin ruptura de simetría.
  - `estrellas`: clases de grado rojo del vértice 0, como `canonical_star` 18..21 en K43.
    El rango sale de las cotas de grado y, si s=t, del intercambio de colores.

Reproducir:

```
cargo test                                                     # 14 tests, ~7 s
cargo test --release -- --include-ignored                      # + casos pesados
cargo test --release --features gpu --test ramsey_gpu_parity -- --ignored
cargo run --release --bin ramsey_bench -- --budget 100000000 --seeds 3 --jobs 8
cargo run --release --bin ramsey_bench -- --knuth 100000 --only 3,6,18
cargo run --release --bin ramsey_bench -- --knuth 2000 --only 5,5,43 --root estrellas
```

## Veredictos

| R(s,t) | K_{R-1} debe ser SAT | K_R debe ser UNSAT |
|---|---|---|
| R(3,3)=6 | SAT con todas las heurísticas | UNSAT con todas; las estrellas quedan vacías (sólo cotas) |
| R(3,4)=9 | SAT con todas | UNSAT con todas, desde vacía y por estrellas |
| R(3,5)=14 | SAT con todas salvo `lex` (se agota el presupuesto) | UNSAT desde vacía (lex+LA: 4 002 transiciones); estrellas vacías |
| R(4,4)=18 | SAT sólo con fail-first (1 de 3 semillas desde vacía, 2 de 3 por estrellas); el testigo es srg(17,8,3,4) = Paley(17) | UNSAT desde vacía (lex+LA: 97 238); estrellas vacías |
| R(3,6)=18 | SAT con lex+LA (2 de 3 semillas), fail-first, neuro/32 y azar/1 (1 de 3) | **No alcanzado**: UNKNOWN con todas (Knuth: ≥ 4,6·10¹⁶ transiciones con la mejor estrategia) |

Ninguna de las 140 filas (360 ejecuciones) da un veredicto que contradiga el valor
conocido, y cada testigo SAT se comprueba también por fuerza bruta. El único hueco es
**R(3,6) K18 UNSAT**, un resultado de 1966 que este DFS no puede reproducir en un tiempo
viable.

## Tamaño del árbol UNSAT (transiciones exactas, mediana de 3 semillas en las políticas)

Agotar un árbol de N nodos cuesta exactamente 2N transiciones (comprobado en
`knuth_estimate_matches_exhausted_trees`). Las políticas añaden una transición por consulta.
`>10⁸` significa que se agotó el presupuesto; en ese caso se da entre paréntesis la estimación
de Knuth cuando la hay.

| caso, raíz | lex | lex+LA | fail-first | neuro/32 | azar/32 | neuro/1 | azar/1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R(3,3) K6, vacía | 1 974 | 38 | 38 | 39 | 75 | 157 | 242 |
| R(3,4) K9, vacía | 29 196 464 | 711 368 | **140 080** | 687 946 | 805 753 | 818 170 | 4 888 952 |
| R(3,4) K9, estrellas | 386 002 | 7 832 | **3 292** | 7 348 | 8 497 | 10 175 | 41 629 |
| R(3,5) K14, vacía | >10⁸ | **4 002** | 1 063 136 | 5 575 | 6 998 | >10⁸ | >10⁸ |
| R(4,4) K18, vacía | >10⁸ (Knuth 7·10²⁸) | **97 238** | 6 079 474 | 312 278 | 200 266 | >10⁸ | >10⁸ |
| R(3,6) K18, vacía | >10⁸ (1,8·10²⁶) | >10⁸ (1,7·10²³) | >10⁸ (6,8·10¹⁹) | >10⁸ | >10⁸ | >10⁸ | >10⁸ |
| R(3,6) K18, estrellas | >2·10⁸ (1,2·10²³) | >2·10⁸ (6,2·10¹⁹) | >2·10⁸ (**4,6·10¹⁶**) | >2·10⁸ | >2·10⁸ | >2·10⁸ | >2·10⁸ |
| R(5,5) K43, estrellas | (1,9·10¹⁴²) | (5,6·10¹²⁷) | (1,1·10⁹⁹) | — | — | — | — |

Las estimaciones de Knuth (100 000 sondas en R(3,6) y 2 000 en K43) tienen colas pesadas. En
K43 el error estándar iguala a la media, así que sólo valen como orden de magnitud. Las
políticas no se pueden estimar así, porque su elección depende de la historia.

## Hallazgos

1. **La referencia es el motor.** La paridad contador a contador con los kernels journal
   permite medir en CPU, sobre cualquier (s,t,n), el mismo árbol que recorre la GPU.
2. **Las cotas de grado hacen el trabajo en 3 de los 5 casos.** En R(3,3), R(3,5) y R(4,4),
   R(s-1,t)+R(s,t-1) ya es R(s,t): las clases de estrella quedan vacías y K_R es UNSAT sin
   búsqueda. Por eso se resuelven también desde la raíz vacía. En K43 no ocurre así: hay 4
   clases (18..21) y la búsqueda es real.
3. **Sin lookahead, el DFS puro no escala ni en casos de juguete.** R(3,4) K9 necesita 2,9·10⁷
   transiciones, y R(4,4) K18 se estima en ~7·10²⁸. El lookahead (cotas y forzado) es lo que
   hace viables los casos pequeños.
4. **La red neuronal es mejor que el azar, pero no mejor que lex+LA.** Con intervalo 1, neuro
   explora 6× menos que azar en R(3,4) K9 (818 k frente a 4,9 M), así que algo aprendió. Pero
   frente a la línea base sin red:
   - R(3,4) K9: neuro/32 queda en −3 % (vacía) y −6 % (estrellas). La variación entre sus
     semillas es inferior al 1 %, pero azar/32 da +13 % y +8 %: es una mejora pequeña y real
     sólo en este caso.
   - R(3,5) K14: +39 %.
   - R(4,4) K18: +221 %, peor incluso que azar/32.
   - neuro/1 empeora siempre (+15 %, +30 %) y no termina R(3,5) K14 ni R(4,4) K18, que lex+LA
     cierra con 4 002 y 97 238 transiciones.

   Con la cadencia por defecto (32), la red sólo decide 1 de cada 32 elecciones, así que el
   árbol es esencialmente el de lex+LA. **No hay evidencia en estos casos de que la red
   reduzca el árbol.** La red fue entrenada sobre K43, así que estos casos están fuera de su
   distribución: el resultado no prueba nada sobre K43, pero tampoco hay ningún dato que
   apoye la reducción del 20 % que menciona el README.
5. **Fail-first no domina.** Reduce el árbol 5× en R(3,4) K9, es la única estrategia que
   encuentra el grafo de R(4,4) en K17 y da la mejor estimación en R(3,6) K18 y en K43. Pero
   empeora 265× en R(3,5) K14 y 63× en R(4,4) K18: no completa la fila de un vértice, que es
   donde las cotas de grado refutan. Además, cada transición le cuesta unas 20 veces más
   checks que a lex+LA.
   Desempatar primero por vecinos comunes (la primera variante probada) daba 63 716 en
   R(3,4) K9, pero 39,6 M en R(3,5) K14; la variante elegida gana en conjunto.
6. **SAT depende de la suerte.** En K_{R-1} el rango entre semillas llega a 10⁴×: por ejemplo,
   fail-first en R(3,5) K13 va de 941 a 1,4 M. Comparar heurísticas en SAT con una sola semilla
   no informa.
7. **Escala.** R(3,6)=18 tiene 153 aristas y su mejor estimación es ~5·10¹⁶ transiciones, unos
   700 años a 2·10⁶ transiciones/s en CPU. K43 se estima en 10⁹⁹–10¹²⁸. Cambiar el orden de
   ramificación mueve decenas de órdenes de magnitud, pero no acerca ninguno de los dos a algo
   ejecutable. Haría falta una ruptura de simetría y una propagación mucho más fuertes que las
   actuales.

## Limitaciones

- La MLP en CPU (f32) puede romper empates de forma distinta que TensorRT (FP16). No se
  ejecutó el kernel neuro real.
- La columna de tiempo es tiempo de CPU por ejecución, con 8 ejecuciones en paralelo. Para
  neuro mide la MLP en CPU, no el coste en GPU. El tamaño del árbol (transiciones) es la
  métrica comparable.
- La heurística fail-first es propia de este banco: el kernel no la tiene.

## Tabla completa

Ejecutada con `ramsey_bench --budget 100000000 --seeds 3 --jobs 8` (las filas azar, en una
segunda pasada con `--strategy azar/32+LA,azar/1+LA`). El presupuesto es por clase de
estrella. En K_R, lex y fail-first usan una sola semilla porque su árbol no depende de ella.
Las columnas decisiones, conflictos, checks y consultas son medianas.

| caso | esperado | raíz | heurística | veredicto | transiciones (mediana) | mín–máx | decisiones | conflictos | checks | consultas | tiempo |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| R(3,3) K5 | SAT | vacía | lex | SAT | 39 | 31–65 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K5 | SAT | vacía | lex+LA | SAT | 11 | 11–11 | 3 | 0 | 10 | 0 | 0.000 s |
| R(3,3) K5 | SAT | vacía | fail-first+LA | SAT | 11 | 11–11 | 3 | 0 | 34 | 0 | 0.000 s |
| R(3,3) K5 | SAT | vacía | neuro/32+LA | SAT | 12 | 12–12 | 3 | 0 | 11 | 1 | 0.000 s |
| R(3,3) K5 | SAT | vacía | azar/32+LA | SAT | 14 | 12–14 | 5 | 1 | 13 | 1 | 0.000 s |
| R(3,3) K5 | SAT | vacía | neuro/1+LA | SAT | 15 | 15–15 | 4 | 0 | 14 | 4 | 0.000 s |
| R(3,3) K5 | SAT | vacía | azar/1+LA | SAT | 15 | 15–19 | 4 | 0 | 14 | 4 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | lex | SAT | 15 | 9–21 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | lex+LA | SAT | 7 | 7–7 | 1 | 0 | 6 | 0 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | fail-first+LA | SAT | 7 | 7–7 | 1 | 0 | 13 | 0 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | neuro/32+LA | SAT | 8 | 8–8 | 1 | 0 | 7 | 1 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | azar/32+LA | SAT | 8 | 8–10 | 1 | 0 | 7 | 1 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | neuro/1+LA | SAT | 8 | 8–8 | 1 | 0 | 7 | 1 | 0.000 s |
| R(3,3) K5 | SAT | estrellas 2..2 | azar/1+LA | SAT | 8 | 8–11 | 1 | 0 | 7 | 1 | 0.000 s |
| R(3,3) K6 | UNSAT | vacía | lex | UNSAT | 1974 | 1974–1974 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | vacía | lex+LA | UNSAT | 38 | 38–38 | 10 | 6 | 24 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | vacía | fail-first+LA | UNSAT | 38 | 38–38 | 10 | 6 | 88 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | vacía | neuro/32+LA | UNSAT | 39 | 39–39 | 10 | 6 | 25 | 1 | 0.000 s |
| R(3,3) K6 | UNSAT | vacía | azar/32+LA | UNSAT | 75 | 39–75 | 22 | 12 | 49 | 1 | 0.000 s |
| R(3,3) K6 | UNSAT | vacía | neuro/1+LA | UNSAT | 157 | 157–157 | 36 | 19 | 106 | 19 | 0.001 s |
| R(3,3) K6 | UNSAT | vacía | azar/1+LA | UNSAT | 242 | 154–288 | 58 | 30 | 165 | 30 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | lex | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | lex+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | fail-first+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | neuro/32+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | azar/32+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | neuro/1+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,3) K6 | UNSAT | estrellas: ninguna (cotas) | azar/1+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,4) K8 | SAT | vacía | lex | SAT | 7943 | 1053–14971 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,4) K8 | SAT | vacía | lex+LA | SAT | 39 | 29–61 | 19 | 2 | 35 | 0 | 0.000 s |
| R(3,4) K8 | SAT | vacía | fail-first+LA | SAT | 29 | 29–29 | 13 | 0 | 245 | 0 | 0.000 s |
| R(3,4) K8 | SAT | vacía | neuro/32+LA | SAT | 40 | 30–90 | 19 | 2 | 36 | 1 | 0.000 s |
| R(3,4) K8 | SAT | vacía | azar/32+LA | SAT | 40 | 30–62 | 19 | 2 | 36 | 1 | 0.000 s |
| R(3,4) K8 | SAT | vacía | neuro/1+LA | SAT | 192 | 188–313 | 46 | 18 | 143 | 31 | 0.001 s |
| R(3,4) K8 | SAT | vacía | azar/1+LA | SAT | 149 | 87–1004 | 36 | 10 | 111 | 28 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | lex | SAT | 3066 | 974–4188 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | lex+LA | SAT | 268 | 104–306 | 64 | 27 | 171 | 0 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | fail-first+LA | SAT | 22 | 22–42 | 9 | 0 | 157 | 0 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | neuro/32+LA | SAT | 257 | 136–303 | 65 | 27 | 167 | 3 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | azar/32+LA | SAT | 303 | 59–369 | 70 | 30 | 193 | 3 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | neuro/1+LA | SAT | 198 | 178–214 | 45 | 19 | 143 | 30 | 0.000 s |
| R(3,4) K8 | SAT | estrellas 2..3 | azar/1+LA | SAT | 220 | 119–495 | 55 | 22 | 159 | 34 | 0.000 s |
| R(3,4) K9 | UNSAT | vacía | lex | UNSAT | 29196464 | 29196464–29196464 | 0 | 0 | 0 | 0 | 0.684 s |
| R(3,4) K9 | UNSAT | vacía | lex+LA | UNSAT | 711368 | 711368–711368 | 160812 | 80407 | 436090 | 0 | 0.017 s |
| R(3,4) K9 | UNSAT | vacía | fail-first+LA | UNSAT | 140080 | 140080–140080 | 16636 | 8319 | 369298 | 0 | 0.035 s |
| R(3,4) K9 | UNSAT | vacía | neuro/32+LA | UNSAT | 687946 | 686083–688739 | 151856 | 75929 | 422270 | 4630 | 0.255 s |
| R(3,4) K9 | UNSAT | vacía | azar/32+LA | UNSAT | 805753 | 797688–806993 | 179326 | 89664 | 495222 | 5454 | 0.187 s |
| R(3,4) K9 | UNSAT | vacía | neuro/1+LA | UNSAT | 818170 | 818170–818170 | 121546 | 60774 | 508219 | 76722 | 3.348 s |
| R(3,4) K9 | UNSAT | vacía | azar/1+LA | UNSAT | 4888952 | 4681929–5193353 | 925386 | 462694 | 3171433 | 528528 | 1.962 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | lex | UNSAT | 386002 | 386002–386002 | 0 | 0 | 0 | 0 | 0.006 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | lex+LA | UNSAT | 7832 | 7832–7832 | 2116 | 1059 | 4974 | 0 | 0.000 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | fail-first+LA | UNSAT | 3292 | 3292–3292 | 392 | 197 | 7781 | 0 | 0.000 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | neuro/32+LA | UNSAT | 7348 | 7300–7362 | 1950 | 976 | 4683 | 60 | 0.002 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | azar/32+LA | UNSAT | 8497 | 8279–10131 | 2258 | 1130 | 5412 | 69 | 0.001 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | neuro/1+LA | UNSAT | 10175 | 10175–10175 | 1696 | 849 | 6467 | 1063 | 0.045 s |
| R(3,4) K9 | UNSAT | estrellas 3..3 | azar/1+LA | UNSAT | 41629 | 37924–42014 | 8186 | 4094 | 27287 | 4759 | 0.011 s |
| R(3,5) K13 | SAT | vacía | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 6.645 s |
| R(3,5) K13 | SAT | vacía | lex+LA | SAT | 703 | 389–39155 | 145 | 59 | 449 | 0 | 0.001 s |
| R(3,5) K13 | SAT | vacía | fail-first+LA | SAT | 1285 | 941–1415943 | 134 | 56 | 10180 | 0 | 0.464 s |
| R(3,5) K13 | SAT | vacía | neuro/32+LA | SAT | 16672 | 266–38294 | 4129 | 2052 | 10490 | 127 | 0.007 s |
| R(3,5) K13 | SAT | vacía | azar/32+LA | SAT | 16192 | 2479–4371774 | 4106 | 2039 | 10236 | 125 | 0.546 s |
| R(3,5) K13 | SAT | vacía | neuro/1+LA | SAT | 11766538 | 2257236–93368849 | 1827260 | 913616 | 7408361 | 1222875 | 258.177 s |
| R(3,5) K13 | SAT | vacía | azar/1+LA | SAT | 62188 | 22803–681187 | 13339 | 6655 | 41543 | 7511 | 0.125 s |
| R(3,5) K13 | SAT | estrellas 4..4 | lex | UNKNOWN/UNKNOWN/SAT | 100000000 | 1776391–100000000 | 0 | 0 | 0 | 0 | 4.761 s |
| R(3,5) K13 | SAT | estrellas 4..4 | lex+LA | SAT | 20307 | 6629–22333 | 4177 | 2075 | 12261 | 0 | 0.001 s |
| R(3,5) K13 | SAT | estrellas 4..4 | fail-first+LA | SAT | 6173 | 643–7199 | 732 | 357 | 21975 | 0 | 0.003 s |
| R(3,5) K13 | SAT | estrellas 4..4 | neuro/32+LA | SAT | 16398 | 6702–17478 | 3565 | 1769 | 10257 | 109 | 0.005 s |
| R(3,5) K13 | SAT | estrellas 4..4 | azar/32+LA | SAT | 22714 | 8191–28073 | 4705 | 2339 | 13800 | 143 | 0.003 s |
| R(3,5) K13 | SAT | estrellas 4..4 | neuro/1+LA | SAT | 5293 | 1034–7228 | 898 | 438 | 3422 | 610 | 0.034 s |
| R(3,5) K13 | SAT | estrellas 4..4 | azar/1+LA | SAT | 804267 | 86561–912655 | 183101 | 91537 | 545400 | 103394 | 0.541 s |
| R(3,5) K14 | UNSAT | vacía | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 2.231 s |
| R(3,5) K14 | UNSAT | vacía | lex+LA | UNSAT | 4002 | 4002–4002 | 988 | 495 | 2495 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | vacía | fail-first+LA | UNSAT | 1063136 | 1063136–1063136 | 219216 | 109609 | 11742565 | 0 | 0.543 s |
| R(3,5) K14 | UNSAT | vacía | neuro/32+LA | UNSAT | 5575 | 5371–7770 | 1386 | 694 | 3502 | 43 | 0.006 s |
| R(3,5) K14 | UNSAT | vacía | azar/32+LA | UNSAT | 6998 | 6706–14779 | 1710 | 856 | 4380 | 52 | 0.001 s |
| R(3,5) K14 | UNSAT | vacía | neuro/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 23107661 | 11553826 | 68011111 | 12914539 | 1461.295 s |
| R(3,5) K14 | UNSAT | vacía | azar/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 25375654 | 12687818 | 69349353 | 13507606 | 59.626 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | lex | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | lex+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | fail-first+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | neuro/32+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | azar/32+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | neuro/1+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,5) K14 | UNSAT | estrellas: ninguna (cotas) | azar/1+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K17 | SAT | vacía | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 7.385 s |
| R(4,4) K17 | SAT | vacía | lex+LA | UNKNOWN | 100000000 | 100000000–100000000 | 29164640 | 14582299 | 64582340 | 0 | 11.292 s |
| R(4,4) K17 | SAT | vacía | fail-first+LA | SAT/UNKNOWN/UNKNOWN | 100000000 | 2683627–100000000 | 8445315 | 4222635 | 648918733 | 0 | 95.524 s |
| R(4,4) K17 | SAT | vacía | neuro/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 28749658 | 14374808 | 64815434 | 881175 | 102.364 s |
| R(4,4) K17 | SAT | vacía | azar/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 23642635 | 11821286 | 62182111 | 721538 | 23.957 s |
| R(4,4) K17 | SAT | vacía | neuro/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 18055037 | 9027501 | 67009777 | 15964484 | 1855.634 s |
| R(4,4) K17 | SAT | vacía | azar/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 24269064 | 12134501 | 68761799 | 13649583 | 58.923 s |
| R(4,4) K17 | SAT | estrellas 8..8 | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 6.974 s |
| R(4,4) K17 | SAT | estrellas 8..8 | lex+LA | UNKNOWN | 100000000 | 100000000–100000000 | 31154477 | 15577222 | 65577254 | 0 | 9.719 s |
| R(4,4) K17 | SAT | estrellas 8..8 | fail-first+LA | SAT/SAT/UNKNOWN | 34794045 | 639939–100000000 | 3044472 | 1522218 | 244301380 | 0 | 65.924 s |
| R(4,4) K17 | SAT | estrellas 8..8 | neuro/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 27004275 | 13502120 | 63914784 | 825268 | 106.805 s |
| R(4,4) K17 | SAT | estrellas 8..8 | azar/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 24601041 | 12300506 | 62676171 | 751267 | 17.308 s |
| R(4,4) K17 | SAT | estrellas 8..8 | neuro/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 19308334 | 9654151 | 67257613 | 15206840 | 1926.894 s |
| R(4,4) K17 | SAT | estrellas 8..8 | azar/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 22628663 | 11314304 | 68163382 | 13698053 | 45.809 s |
| R(4,4) K18 | UNSAT | vacía | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 2.708 s |
| R(4,4) K18 | UNSAT | vacía | lex+LA | UNSAT | 97238 | 97238–97238 | 25738 | 12870 | 61488 | 0 | 0.003 s |
| R(4,4) K18 | UNSAT | vacía | fail-first+LA | UNSAT | 6079474 | 6079474–6079474 | 1174606 | 587304 | 80172514 | 0 | 4.946 s |
| R(4,4) K18 | UNSAT | vacía | neuro/32+LA | UNSAT | 312278 | 234751–357434 | 84216 | 42109 | 199535 | 2576 | 0.271 s |
| R(4,4) K18 | UNSAT | vacía | azar/32+LA | UNSAT | 200266 | 175421–236682 | 53124 | 26563 | 127504 | 1618 | 0.017 s |
| R(4,4) K18 | UNSAT | vacía | neuro/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 23784756 | 11892365 | 68506576 | 13228368 | 1731.485 s |
| R(4,4) K18 | UNSAT | vacía | azar/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 26285875 | 13142908 | 70632330 | 14978754 | 41.195 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | lex | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | lex+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | fail-first+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | neuro/32+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | azar/32+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | neuro/1+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(4,4) K18 | UNSAT | estrellas: ninguna (cotas) | azar/1+LA | UNSAT | 0 | 0–0 | 0 | 0 | 0 | 0 | 0.000 s |
| R(3,6) K17 | SAT | vacía | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 9.368 s |
| R(3,6) K17 | SAT | vacía | lex+LA | SAT/SAT/UNKNOWN | 73856221 | 60630999–100000000 | 20410694 | 10205318 | 46722530 | 0 | 11.298 s |
| R(3,6) K17 | SAT | vacía | fail-first+LA | SAT/UNKNOWN/UNKNOWN | 100000000 | 6191917–100000000 | 9749021 | 4874491 | 454265365 | 0 | 99.045 s |
| R(3,6) K17 | SAT | vacía | neuro/32+LA | UNKNOWN/SAT/UNKNOWN | 100000000 | 80061497–100000000 | 26813779 | 13406861 | 63715319 | 820354 | 105.937 s |
| R(3,6) K17 | SAT | vacía | azar/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 28964643 | 14482300 | 64925035 | 885366 | 31.112 s |
| R(3,6) K17 | SAT | vacía | neuro/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 20858600 | 10429280 | 67381178 | 13851456 | 1660.829 s |
| R(3,6) K17 | SAT | vacía | azar/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 22129293 | 11064624 | 67774515 | 13419680 | 44.376 s |
| R(3,6) K17 | SAT | estrellas 3..5 | lex | UNKNOWN | 300000000 | 300000000–300000000 | 0 | 0 | 0 | 0 | 28.808 s |
| R(3,6) K17 | SAT | estrellas 3..5 | lex+LA | UNKNOWN | 300000000 | 300000000–300000000 | 92597060 | 46298485 | 196298612 | 0 | 41.987 s |
| R(3,6) K17 | SAT | estrellas 3..5 | fail-first+LA | SAT/UNKNOWN/UNKNOWN | 300000000 | 200010927–300000000 | 29692274 | 14846090 | 1589570102 | 0 | 484.621 s |
| R(3,6) K17 | SAT | estrellas 3..5 | neuro/32+LA | UNKNOWN | 300000000 | 300000000–300000000 | 88889487 | 44444697 | 195803606 | 2717553 | 368.880 s |
| R(3,6) K17 | SAT | estrellas 3..5 | azar/32+LA | UNKNOWN | 300000000 | 300000000–300000000 | 81773203 | 40886540 | 192134843 | 2496345 | 88.999 s |
| R(3,6) K17 | SAT | estrellas 3..5 | neuro/1+LA | UNKNOWN | 300000000 | 300000000–300000000 | 55272713 | 27636304 | 198526368 | 41779876 | 7225.279 s |
| R(3,6) K17 | SAT | estrellas 3..5 | azar/1+LA | SAT/UNKNOWN/UNKNOWN | 300000000 | 202425741–300000000 | 60905344 | 30452612 | 198453786 | 36002063 | 135.652 s |
| R(3,6) K18 | UNSAT | vacía | lex | UNKNOWN | 100000000 | 100000000–100000000 | 0 | 0 | 0 | 0 | 3.334 s |
| R(3,6) K18 | UNSAT | vacía | lex+LA | UNKNOWN | 100000000 | 100000000–100000000 | 26646501 | 13323226 | 63323290 | 0 | 5.386 s |
| R(3,6) K18 | UNSAT | vacía | fail-first+LA | UNKNOWN | 100000000 | 100000000–100000000 | 10064530 | 5032245 | 570354476 | 0 | 67.136 s |
| R(3,6) K18 | UNSAT | vacía | neuro/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 28666015 | 14332981 | 64770310 | 874543 | 129.464 s |
| R(3,6) K18 | UNSAT | vacía | azar/32+LA | UNKNOWN | 100000000 | 100000000–100000000 | 28573689 | 14286821 | 64721400 | 869038 | 19.433 s |
| R(3,6) K18 | UNSAT | vacía | neuro/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 17354670 | 8677311 | 65602797 | 13850850 | 1986.832 s |
| R(3,6) K18 | UNSAT | vacía | azar/1+LA | UNKNOWN | 100000000 | 100000000–100000000 | 22413606 | 11206774 | 67968654 | 13523641 | 45.383 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | lex | UNKNOWN | 200000000 | 200000000–200000000 | 0 | 0 | 0 | 0 | 7.471 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | lex+LA | UNKNOWN | 200000000 | 200000000–200000000 | 50358314 | 25179122 | 125179218 | 0 | 10.911 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | fail-first+LA | UNKNOWN | 200000000 | 200000000–200000000 | 19638661 | 9819303 | 1246540194 | 0 | 132.635 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | neuro/32+LA | UNKNOWN | 200000000 | 200000000–200000000 | 48850013 | 24424963 | 125171391 | 1492648 | 244.523 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | azar/32+LA | UNKNOWN | 200000000 | 200000000–200000000 | 48797206 | 24398555 | 125143436 | 1489560 | 38.986 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | neuro/1+LA | UNKNOWN | 200000000 | 200000000–200000000 | 36180745 | 18090330 | 132110154 | 28039459 | 4253.015 s |
| R(3,6) K18 | UNSAT | estrellas 4..5 | azar/1+LA | UNKNOWN | 200000000 | 200000000–200000000 | 46075992 | 23037952 | 137114278 | 28152446 | 93.297 s |
