# `sat/`: prototipo de demostración certificada para Ramsey (s, t, n)

Núcleo SAT con prueba verificable para enunciados del tipo «toda 2-coloración
de K_n contiene un K_s rojo o un K_t azul», pensado como base de un futuro
ataque a R(5,5) = 43 (K43). No toca el motor GPU del repositorio.

Flujo: `ramsey_cnf.py` genera la CNF → CaDiCaL la resuelve y escribe una prueba
LRAT → `lrat-check` comprueba la prueba LRAT y `drat-trim` comprueba los mismos
lemas reescritos como DRAT. Si sale SAT, el modelo se comprueba directamente
sobre la coloración, sin pasar por la CNF.

**Resultado en una frase:** el núcleo funciona y está validado en todos los
casos con respuesta conocida, pero este método (ruptura de simetría estática más
cube-and-conquer ingenuo) **no está ni cerca** de K43: la cota inferior medida
es ≥ 10^13 s (≈ 5·10^5 años-CPU) solo para la clase d=18 cortando a 60 aristas,
y la única estimación con todas las hojas resueltas es de 10^44–10^59 s por
clase.

- **Validación**: R(3,3)=6, R(3,4)=9, R(3,5)=14 y R(4,4)=18 dan UNSAT en K_R con
  prueba LRAT y DRAT verificadas, y SAT en K_{R−1} con el modelo comprobado. Como
  extra, R(3,6)=18 y R(3,7)=23 sin ningún lema (R(3,7,23): 138 s de solver,
  404 MB de LRAT, ambas comprobaciones OK).
- **Ruptura de simetría**: hay un argumento de corrección (lex-leader sobre el
  grupo que fija la estrella de la raíz) y una auditoría exhaustiva que, para
  n ≤ 8, cubre todas las clases de isomorfismo.
- **K43**: el árbol de cubos tiene ~10^56–10^61 nodos por clase, y las hojas
  difíciles (> 30 min en las probadas) se concentran donde está la masa del
  árbol. Además, el método todavía no reproduce el lema R(4,5)=25 del que depende
  la reducción a d ∈ {18,19,20}.

## Ficheros

| fichero | qué hace |
|---|---|
| `ramsey_cnf.py` | generador CNF (≈300 líneas, solo biblioteca estándar) y comprobador semántico de coloraciones |
| `common.py` | rutas y partición en clases de raíz (`root_classes`) |
| `certify.py` | resuelve y certifica casos (modos `plain`, `classes`, `lemma`) |
| `test_symmetry.py` | auditoría: la ruptura de simetría no pierde clases de isomorfismo (n ≤ 8) |
| `test_encodings.py` | prueba exhaustiva de las codificaciones de cardinalidad y lex |
| `cube_proofs.py` | resuelve cubos sueltos de K43 con LRAT y los verifica (tamaño de prueba por cubo) |
| `report.py` | regenera las tablas de este README a partir de `results/` |
| `graphs.py` | isomorfismo de grafos pequeños (solo para la auditoría) |
| `estimate.py`, `src/cube_estimate.cpp` | estimación del coste cube-and-conquer (Knuth y población) |
| `src/enum_models.cpp` | enumera modelos proyectados en las aristas (para la auditoría) |
| `build_tools.sh` | descarga y compila CaDiCaL y drat-trim (versiones fijadas) y `bin/` |
| `results/` | salidas JSON/texto de todas las ejecuciones de este README |

`tools/`, `bin/` y `work/` (CNF y pruebas temporales) están en `.gitignore`.

## Reproducir

```bash
bash sat/build_tools.sh
```

```bash
cd sat && python test_encodings.py && python test_symmetry.py 8 && python certify.py
```

```bash
cd sat && python certify.py 3,6,18 3,7,23 --modes classes,lemma --timeout 3600 --out results/scaling.json
```

```bash
cd sat && python estimate.py 5 5 43 18 --window 18 24 --k 861 --population 500 --leaf-time 0.001
```

```bash
cd sat && python estimate.py 5 5 43 18 --window 18 24 --k 60 --population 20 --leaf-time 60 --seed 101
```

```bash
cd sat && python report.py
```

El primer bloque ejecuta las auditorías y la validación (~45 min, casi todo
porque el modo `plain` agota sus 20 min en dos casos). El segundo reproduce la
escalabilidad. El tercero calcula el perfil del árbol de K43 de una clase
(~30 min) y el cuarto mide la dureza de las hojas a una profundidad (≤ 20 min). `report.py` regenera las
tablas de este README.

`build_tools.sh` hace `git clone` de `arminbiere/cadical` (tag `rel-3.0.1`,
commit `c60730422e75`) y `marijnheule/drat-trim` (commit `2e3b2dc0ecf9`). En
MinGW, drat-trim necesita `-Dgetc_unlocked=getc`, y el binario de CaDiCaL no
admite `-t`, así que `certify.py` aplica el límite de tiempo desde Python.

## Codificación

- **Variables**: `x(i,j)` para `0 ≤ i < j < n`, numeradas fila a fila
  (`(0,1),(0,2),…`); verdadero = rojo. Las auxiliares van después.
- **F (sección `ramsey`)**: una cláusula `∨ ¬x` por cada s-subconjunto (no hay K_s
  rojo) y una `∨ x` por cada t-subconjunto (no hay K_t azul). Es exactamente el
  enunciado; todo lo demás se añade detrás, y cada sección lleva su rango de
  cláusulas en la cabecera (`c section … clauses a..b`) para poder auditarla.
- **Estrella (`star`)**: unitarias; la raíz 0 es roja con `1..d` y azul con `d+1..n-1`.
- **Ventana de grado (`degree`)**: `lo ≤ grado_rojo(v) ≤ hi` para todo `v ≠ 0`.
  Por defecto se usa un totalizador (Bailleux–Boufkhad) con las dos direcciones,
  así que sus salidas quedan *definidas* (`r_m ⇔ #rojos ≥ m`). La alternativa es
  el contador secuencial de Sinz (`--card seq`).
- **Lex (`lex`)**: ver la sección siguiente. Cada `a ≤_lex b` usa tres cláusulas
  de 3 literales por posición, con una variable «prefijo igual».

## Qué se demuestra y en qué hay que confiar

Para cerrar «R(s,t) ≤ n» hacen falta cuatro piezas. Solo la tercera la verifica
una máquina de forma independiente; las demás son argumentos que hay que auditar.

1. **Partición en clases** (`common.root_classes`, argumento en papel). Sea `g(v)`
   el grado rojo y `m = min_v min(g(v), n−1−g(v))`. Se renombran los vértices para
   que la raíz 0 alcance ese mínimo; así todos los grados caen en `[m, n−1−m]` y la
   raíz tiene grado rojo `m` o `n−1−m`. Si `s = t`, intercambiar colores es una
   simetría y basta con grado `m`. La clase `m = (n−1)/2` con `n` impar y `m` impar
   se descarta por el lema del apretón de manos: no existe un grafo regular de
   grado impar con un número impar de vértices. Es mejor no pedirle esto a CDCL,
   porque los argumentos de paridad son exponenciales para resolución. Renombrar
   vértices es correcto porque F es invariante bajo Sym(n).
2. **Corrección de la ruptura de simetría** (argumento en la sección siguiente,
   auditado empíricamente).
3. **UNSAT de cada CNF de clase**: prueba LRAT de CaDiCaL comprobada con
   `lrat-check`, y los mismos lemas en DRAT con `drat-trim`. No se confía en el
   solver.
4. **Generador**: `ramsey_cnf.py` es corto y determinista; la sección `ramsey` es
   literalmente `itertools.combinations`. Los modelos SAT se comprueban con
   `check_coloring`, que busca cliques monocromáticos por backtracking, sin la CNF.

**Lemas externos.** El modo `lemma` restringe los grados con números de Ramsey
menores: `n − R(s,t−1) ≤ grado_rojo ≤ R(s−1,t) − 1`. Esa cota es un hecho
matemático, pero *no* lo comprueba la prueba LRAT. Para K43 da `[18, 24]` a partir
de R(4,5) = 25 (McKay–Radziszowski 1995, por ordenador y sin certificado
DRAT/LRAT). De ahí salen las clases d ∈ {18, 19, 20}; la 21 cae por paridad
(43·21 es impar). Sin ese lema habría que refutar también d ≤ 17, que contiene
como subproblema un vecindario azul de ≥ 25 vértices sin K4 azul ni K5 rojo, es
decir, volver a demostrar R(4,5) ≤ 25. Una demostración de K43 por esta vía depende,
por tanto, de R(4,5) = 25, salvo que ese resultado se certifique aparte.

## Por qué la ruptura de simetría es correcta

Se fija una clase: raíz con grado rojo `d` y ventana `[lo, hi]`. Sean `R = {1..d}`
y `B = {d+1..n−1}`, y sea `G' = Sym(R) × Sym(B)`, las permutaciones de vértices
que fijan 0 y respetan los dos vecindarios. Sea `S` el conjunto de coloraciones
(proyectadas a las aristas) que cumplen F, la estrella y la ventana.

- **S es G'-invariante.** F lo es bajo Sym(n). La estrella se conserva porque
  `σ ∈ G'` fija 0 y lleva R a R. La ventana se conserva porque σ permuta los grados.
  Las auxiliares del totalizador no estorban: para cada asignación de aristas,
  existe una extensión a las auxiliares exactamente cuando los grados están en la
  ventana (extensión definicional).
- **Lex-leader** (Crawford–Ginsberg–Luks–Roy 1996). Si se fija un orden total de
  las variables, el mínimo lexicográfico de cada órbita `G'·x ⊆ S` cumple
  `x ≤_lex σ(x)` para *todo* `σ ∈ G'` a la vez. Añadir cualquier subconjunto de
  esas restricciones, todas con el mismo orden y la misma dirección, conserva la
  satisfacibilidad: si S ≠ ∅, el representante mínimo sobrevive.
- **Reducción a filas** (Codish, Miller, Prosser, Stuckey, *Constraints* 2019).
  Con el orden fila a fila y `σ = (i j)`, `i < j`, `x` y `σ(x)` solo difieren en
  las aristas que tocan i o j. La primera posición candidata es `(a,i)` con `a < i`,
  que compara `x(a,i)` con `x(a,j)`; después vienen las de la fila i, `(i,b)` con
  `b ≠ j`, que comparan `x(i,b)` con `x(j,b)`. Si todas coinciden, las demás
  posiciones también coinciden. Por eso
  `x ≤_lex (i j)x ⇔ fila_i∖{i,j} ≤_lex fila_j∖{i,j}`, comparando columnas en orden
  creciente. Es lo que emite `lex_leq`.
- Solo se usan pares `i < j` del **mismo** bloque (R o B), porque ahí `(i j) ∈ G'`.
  Un par de bloques distintos no es simetría de la clase, porque rompe la estrella.
  El control negativo de `test_symmetry.py` añade un solo par así y pierde clases
  de isomorfismo.
- `--lex full` (todos los pares de V) solo es correcto sin estrella, y el generador
  rechaza esa combinación. `--lex-adjacent` (solo `(i,i+1)`) es un subconjunto del
  anterior, así que también es correcto, aunque más débil.

**Auditoría empírica** (`results/test_symmetry.txt`). Sin cláusulas de Ramsey, las
CNF de clase describen grafos etiquetados. La auditoría enumera todos sus modelos
con `enum_models`, los agrupa por isomorfismo y exige que la unión de las clases
cubra todos los grafos de n vértices: OEIS A000088, o `(A000088 + A000171)/2`
módulo complemento cuando hay intercambio de colores.

| n | lex=root (con intercambio de colores) | lex=root (sin intercambio) | lex=full sin raíz | control negativo |
|---|---|---|---|---|
| 6 | 78 / 78 | 156 / 156 | 156 / 156 | 34 / 78 |
| 7 | 522 / 522 | 1044 / 1044 | 1044 / 1044 | 156 / 522 |
| 8 | 6178 / 6178 | 12346 / 12346 | 12346 / 12346 | 1044 / 6178 |

Cada celda da las clases cubiertas sobre las esperadas; se comprobó lo mismo para
n = 4 y 5. Las variantes `--lex-adjacent` y `--card seq` también cubren todas las
clases. Además, `test_encodings.py` compara de forma exhaustiva los modelos del
totalizador, del contador secuencial (m ≤ 7, todo `lo ≤ hi`) y de `lex_leq`
(longitud ≤ 5) con su semántica: 243 comprobaciones, 0 fallos
(`results/test_encodings.txt`).

**Cómo certificarlo formalmente** (para que la base de confianza se reduzca al
generador de F y a un verificador):

- **VeriPB**, con la regla de dominancia (Bogaerts, Gocht, McCreesh, Nordström,
  *JAIR* 2023), deriva `x ≤_lex σ(x)` dando σ como testigo. La traducción a CNF de
  la ventana también se puede certificar (Gocht, Martins, Nordström, Oertel,
  SAT 2022), y CaDiCaL 3 trae un trazador `--veripb`. La cadena completa (F +
  restricciones de clase en PB → lex por dominancia → codificación → refutación
  CDCL) la podría comprobar VeriPB y, al final, CakePB, que está verificado en
  CakeML. El generador tendría que emitir los pasos de dominancia; aquí no está
  implementado.
- **DRAT/PR.** Heule, Hunt y Wetzler (CADE 2015) muestran cómo expresar la ruptura
  de simetría dentro de una prueba DRAT para simetrías involutivas (las
  transposiciones lo son). Las cláusulas PR (Heule–Kiesl–Biere) permiten pasos
  parecidos sin variables nuevas y se comprueban con `dpr-trim`.
- **SAT Modulo Symmetries** (Kirchweger–Szeider) hace la ruptura dinámica completa
  (comprueba que la matriz es canónica) y se ha usado en problemas tipo Ramsey. Los
  trabajos SAT+CAS sobre R(3,8) y R(3,9) (Li, Bright, Ganesh y otros) producen
  certificados verificables de ese tipo de poda. Conviene revisar el estado de sus
  herramientas de prueba antes de depender de ellas.
- **La partición en clases** (pieza 1) se puede formalizar en Lean, como hizo el
  proyecto del hexágono vacío (Subercaseaux et al., ITP 2024) con su codificación.

## Validación con respuesta conocida

`python certify.py` (modos `classes` y `plain`). En `classes` no se usa ningún
lema: la clase de raíz se recorre para todo `m`. Las pruebas se cuentan en bytes
de LRAT en texto.

| caso | modo | clases | veredicto | esperado | t solver | prueba LRAT | lrat-check | drat-trim |
|---|---|---|---|---|---|---|---|---|
| (3,3,5) | classes | 3 | SAT | SAT | 0.09 s | 0.00 MB | 2/2 | 2/2 (0.3 s) |
| (3,3,6) | classes | 3 | UNSAT | UNSAT | 0.07 s | 0.00 MB | 3/3 | 3/3 (0.4 s) |
| (3,3,6) | plain | 1 | UNSAT | UNSAT | 0.03 s | 0.00 MB | 1/1 | 1/1 (0.1 s) |
| (3,4,8) | classes | 8 | SAT | SAT | 0.22 s | 0.01 MB | 6/6 | 6/6 (0.8 s) |
| (3,4,9) | classes | 9 | UNSAT | UNSAT | 0.25 s | 0.04 MB | 9/9 | 9/9 (1.2 s) |
| (3,4,9) | plain | 1 | UNSAT | UNSAT | 0.19 s | 0.97 MB | 1/1 | 1/1 (0.4 s) |
| (3,5,13) | classes | 13 | SAT | SAT | 0.44 s | 0.28 MB | 12/12 | 12/12 (1.9 s) |
| (3,5,14) | classes | 14 | UNSAT | UNSAT | 0.59 s | 0.59 MB | 14/14 | 14/14 (2.2 s) |
| (3,5,14) | plain | 1 | sin decidir (TIMEOUT 1200 s) | UNSAT | – | – | – | – |
| (4,4,17) | classes | 9 | SAT | SAT | 5.89 s | 6.16 MB | 8/8 | 8/8 (9.6 s) |
| (4,4,18) | classes | 9 | UNSAT | UNSAT | 4.87 s | 6.14 MB | 9/9 | 9/9 (8.6 s) |
| (4,4,18) | plain | 1 | sin decidir (TIMEOUT 1200 s) | UNSAT | – | – | – | – |

- Todos los casos dan el veredicto esperado en modo `classes`. Los UNSAT tienen
  todas sus clases refutadas con LRAT comprobado por `lrat-check` y DRAT
  comprobado por `drat-trim`. En los SAT hay al menos una clase con un modelo que
  `check_coloring` valida; las demás clases de esos casos también llevan prueba,
  y por eso aparecen columnas «k/k». En K17 sale el grafo de Paley 8-regular
  (clase d = 8).
- El modo `plain` (F sola) encuentra rápido los modelos SAT y refuta los casos
  pequeños, pero **no termina en 20 min ni R(3,5,14) ni R(4,4,18)**. Sin ruptura
  de simetría el método no escala ni siquiera a estos casos.
- Los tiempos se midieron con 10–15 procesos a la vez en un portátil de 16 hilos;
  son orientativos.

## Escalabilidad

Instancias justas (n = R(s,t)), todas certificadas salvo donde se indica
(`results/validation.json`, `scaling_a.json`, `scaling_b.json`, `r45_25/`):

| caso | aristas | modo | t solver | prueba LRAT | t de comprobación (lrat-check + drat-trim) |
|---|---|---|---|---|---|
| (3,3,6) | 15 | classes | 0.07 s | 266 B | 0.4 s |
| (3,4,9) | 36 | classes | 0.25 s | 0.04 MB | 1.2 s |
| (3,5,14) | 91 | classes | 0.59 s | 0.59 MB | 2.2 s |
| (4,4,18) | 153 | classes | 4.9 s | 6.1 MB | 8.6 s |
| (3,6,18) | 153 | classes / lemma | 5.1 s / 0.52 s | 8.2 MB / 1.6 MB | 15.3 s / 1.9 s |
| (3,7,23) | 253 | classes / lemma | 138 s / 50 s | 404 MB / 147 MB | 253 s / 64 s |
| (4,5,25) | 300 | lemma, **sin prueba** | d=12: 47 s; las otras 6 clases (d = 7–11, 13): **TIMEOUT a 1 h** | – | – |

Por encima de R, el coste apenas sube: (4,4,19) 3.7 s, (4,4,20) 4.5 s,
(3,4,10/11) < 1 s. Lo que manda es el tamaño del caso justo. Observaciones:

- El coste crece muy deprisa con R(s,t): de (3,6,18) a (3,7,23) es ×27 en tiempo
  y ×50 en prueba por solo 100 aristas más. Ajustar una exponencial entre esos
  dos puntos da ≈ ×1.034 por arista; llevarla a las 903 aristas de K43 daría
  ~10^11–10^12 s. Esa cifra es solo una referencia ingenua: la familia (3,t)
  tiene cláusulas de 3 literales y vecindarios triviales, y es mucho más fácil
  por arista que (5,5); el estimador de árbol de abajo es la cifra relevante.
- Ritmo de prueba ≈ 1.2–3 MB de LRAT por segundo de solver, y comprobación
  ≈ 1–2× el tiempo del solver (lrat-check + drat-trim).
- El lema de grados (modo `lemma`) ahorra 3× en (3,7,23). En (4,5,25), con el
  lema [7,13] (de R(3,5) = 14 y R(4,4) = 18), solo cae la clase 12-regular; las
  demás superan 1 h cada una. **Es decir, este método aún no reproduce
  R(4,5) = 25, el lema del que depende la reducción de K43 a d ∈ {18,19,20}.**

## Estimación de coste para K43 (R(5,5), clases d = 18, 19, 20)

CNF de cada clase: 903 variables de arista + ~26 000 auxiliares y ~2.06·10^6
cláusulas (90 MB); el 93 % son las 2·C(43,5) cláusulas de F. Se genera en ~12 s.

### Método

1. **Árbol de cubos.** Las aristas se recorren en orden estático: primero las
   internas del vecindario rojo de la raíz, después las del azul y al final las
   cruzadas. Una variable ya implicada por propagación unitaria (UP) no ramifica,
   y un valor cuya UP da conflicto se poda (literal fallido de profundidad 1).
   Es un cubeador lookahead elemental. La UP está implementada sin estado en
   `cube_estimate.cpp`, porque `CaDiCaL::propagate()` aprende unidades entre
   llamadas.
2. **N(j)**, el número de nodos vivos con j aristas fijadas, se estima con una
   población de P nodos por nivel y remuestreo sistemático («go with the
   winners»), un estimador insesgado. Calibración frente al árbol exacto:

   | instancia | j | exacto | población |
   |---|---|---|---|
   | R(4,4,18) d=8 | 8 / 14 / 20 / 30 | 5 / 14 / 616 / 53148 cubos | 5 / 14 / 657 / 51152 (P=50) |
   | R(3,7,23) lemma d=5 | 15 | 4944 cubos, 1105 s | 4689 cubos, 1402 s (P=40) |

   El estimador clásico de Knuth (un camino aleatorio) subestima mucho por la
   cola pesada: en d=18, j=60 da 10^9.4–10^11.3 según la semilla, frente a 10^12.1
   con la población.
3. **Dureza de las hojas.** Las hojas a profundidad j se resuelven con CaDiCaL
   (copia fresca del solver, límite T). Se muestrean de dos formas: (a) los nodos
   de la población, uniformes por nodo pero correlacionados porque comparten
   antepasados; (b) caminos de Knuth independientes, con peso 2^ramas. El coste
   del corte a profundidad j es Σ peso·t. Si alguna hoja agota el tiempo, la
   cifra es una **cota inferior**.

### Tamaño del árbol (`results/estimates/*_k861_pop*.json`)

| d | ventana | P | prof. máx. viva | log10 N(153) | log10 N(429) | log10 Σ_j N(j) |
|---|---|---|---|---|---|---|
| 18 | [18, 24] | 500 | 269 | 37.4 | extinta | 60.6 |
| 19 | [19, 23] | 500 | 249 | 35.3 | extinta | 56.5 |
| 19 | [19, 23] | 1000 | 254 | 35.3 | extinta | 56.2 |
| 20 | [20, 22] | 500 | 263 | 34.5 | extinta | 58.6 |

N(j) crece unas 10^0.2 por arista fijada. El árbol podado por UP se extingue
entre las aristas 250 y 270, en las primeras filas del vecindario azul (el
vecindario rojo de d=18 ocupa las aristas 1–153). Por tanto, un DPLL sin
aprendizaje con estas podas recorrería ~10^56–10^61 nodos por clase. Son entre 10^20
y 10^30 veces menos que la estimación de Knuth del DFS de GPU (10^80–10^90),
pero sigue fuera de alcance. La réplica con P=1000 coincide con la de P=500.

### Dureza de las hojas y coste por profundidad de corte

«≥» significa que alguna hoja agotó el tiempo (cota inferior); «≈», que se
resolvieron todas. La tabla se regenera con `python report.py k43`.

| d | j (aristas fijadas) | muestreo | T hoja | log10 N(j) | UNSAT / TIMEOUT | media resueltas | log10 coste (s) |
|---|---|---|---|---|---|---|---|
| 18 | 20 | población P=24 (s41) | 60 s | 3.8 | 12 / 12 | 3.46 s | ≥ 5.3 |
| 18 | 20 | Knuth 60 caminos (s71) | 60 s | 3.3 | 59 / 1 | 0.56 s | ≥ 4.5 |
| 18 | 40 | población P=24 (s41) | 60 s | 8.0 | 10 / 14 | 0.92 s | ≥ 9.5 |
| 18 | 60 | población P=24 (s41) | 60 s | 11.7 | 10 / 14 | 0.32 s | ≥ 13.3 |
| 18 | 60 | Knuth 2500 caminos (s72) | 60 s | 11.3 | 43 / 0 | 0.26 s | ≈ 10.7 |
| 18 | 60 | 6 poblaciones P=20 (s101–106) | 60 s | 11.9 (media) | 76 / 34 | 0.20–0.30 s | ≥ 13.2 (media 1.5·10^13, EE 1.1·10^13) |
| 18 | 60 | población P=6 (s61) | 600 s | 10.3 | 6 / 0 | 0.38 s | ≈ 9.9 |
| 18 | 120 | población P=60 (s51) | 20 s | 28.4 | 5 / 55 | 9.44 s | ≥ 29.7 |
| 18 | 180 | población P=60 (s51) | 20 s | 45.2 | 21 / 39 | 2.79 s | ≥ 46.3 |
| 18 | 240 | población P=60 (s51) | 20 s | 59.2 | 59 / 0 | 0.22 s | ≈ 58.6 |
| 19 | 60 | población P=24 (s41) | 60 s | 11.2 | 24 / 0 | 0.24 s | ≈ 10.5 |
| 19 | 220 | población P=60 (s51) | 20 s | 52.7 | 60 / 0 | 0.28 s | ≈ 52.1 |
| 20 | 60 | población P=24 (s41) | 60 s | 11.5 | 13 / 11 | 2.59 s | ≥ 13.0 |
| 20 | 200 | población P=200 (s52) | 20 s | 44.6 | 199 / 1 | 0.75 s | ≥ 44.5 |

Además:

- **Dos cubos difíciles** de d=18, uno con j=20 y otro con j=60, extraídos de los
  caminos de Knuth: **sin resolver tras 1800 s**, con ~7·10^6 conflictos cada uno
  (`results/hard_d18_k*.txt`).
- **Prueba por cubo fácil** (d=18, j=60, 4 cubos): ~0.32 MB de LRAT (~6000
  lemas), verificados por lrat-check y drat-trim. Cada verificación tarda 8–19 s,
  casi todo en leer los 90 MB de la CNF (`results/cube_proofs_d18_k60.json`).

### Lectura

- **Las hojas son bimodales.** O se refutan casi solo con propagación
  (0.2–0.4 s, que son sobre todo el coste fijo de copiar 2·10^6 cláusulas), o
  CaDiCaL no puede con ellas: las dos probadas siguen abiertas a los 30 min. Las
  difíciles están en la parte frondosa del árbol (el cubo difícil de j=60 tenía
  peso 2^43; los fáciles, 2^22–2^33). Por eso los caminos de Knuth casi nunca las
  ven, mientras que las poblaciones, que muestrean uniformemente por nodo, dan
  un 0–58 % de hojas difíciles según la réplica. Las réplicas independientes
  son la cifra fiable: a j=60, ≥ 1.5·10^13 s de media.
- **Ninguna profundidad de corte sale barata.** Al cortar poco profundo
  (j ≤ 60), la masa del árbol está en cubos que no se resuelven en minutos. Si
  esos cubos de j=60 costaran solo los 30 min ya observados, d=18 necesitaría
  ≥ 0.3 · 8.7·10^11 · 1800 s ≈ 5·10^14 s ≈ 1.5·10^7 años-CPU. A profundidad
  media (j = 120–180), entre el 65 % y el 92 % de las hojas siguen siendo
  difíciles. Solo al cortar hondo (j ≈ 200–240) se resuelven todas, pero
  entonces hay 10^44–10^59 cubos.
- **Estimación para K43 con este método**: por clase, entre ≥ 10^13 s (cota
  medida a j=60 en d=18 y d=20) y ≈ 10^44.5 (d=20), 10^52.1 (d=19) y 10^58.6 s
  (d=18) en el corte más superficial en el que se resolvieron todas las hojas
  muestreadas. Las pruebas, a ~0.3 MB por cubo fácil, suman ≥ 3·10^17 B solo con
  los cubos fáciles de j=60, y 10^50 B o más en el corte profundo. Como
  referencia, Schur 5 costó unos 14 años-CPU y unos 2 PB de prueba DRAT. K43 por
  esta vía está como mínimo 4–5 órdenes de magnitud más allá, y lo más probable
  es que sean bastantes más.
- **Qué tendría que cambiar** (en orden de impacto esperado):
  1. ruptura de simetría completa o rechazo de isomorfos en los vecindarios:
     generar el vecindario azul (24 vértices en d=18) y el rojo como grafos
     (4,5,k) módulo isomorfismo, a partir de catálogos ya enumerados por McKay y
     Radziszowski, y resolver solo el «pegado», como en Angeltveit–McKay
     (R(5,5) ≤ 48 y después ≤ 46);
  2. cotas lineales sobre el número de aristas y de K4 por vecindario, que es lo
     que hace viable ese pegado;
  3. un cubeador lookahead en vez del orden estático;
  4. certificar R(4,5) = 25 por la misma vía, porque la reducción a
     d ∈ {18,19,20} depende de ese lema, y aquí ni siquiera R(4,5,25) cae en
     1 h por clase.

## Limitaciones y siguientes pasos

- **Base de confianza no mecanizada**: la partición en clases (raíz, intercambio
  de colores, paridad), el argumento lex-leader y, para K43, el lema
  R(4,5) = 25. Las vías para mecanizarlos se describen más arriba (VeriPB con
  dominancia y CakePB, DRAT/PR, Lean). Ninguna está implementada aquí.
- **La ruptura de simetría estática es incompleta**: lex por transposiciones deja
  varios representantes etiquetados por clase de isomorfismo, y cada vez más
  según crece n: 1.5 con n = 6, 2.1 con n = 7 y 2.8 con n = 8 (17161 modelos para
  6178 clases). En K43, con vecindarios de 18–24 vértices, el sobrante se
  multiplica en los dos bloques. Lo siguiente sería una ruptura completa: SMS, o
  generar los vecindarios módulo isomorfismo (catálogos de grafos (4,5,k) de
  McKay) y pegarlos, como en Angeltveit–McKay.
- **Cubos ingenuos**: el orden de variables es estático y la poda solo usa UP con
  literales fallidos de profundidad 1. Un cubeador lookahead (march_cu, o
  `generate_cubes` de CaDiCaL) produciría árboles más pequeños; no se descargó
  march_cu (el usuario no lo autorizó) y no se probó `generate_cubes`.
- **El estimador** es insesgado (comprobado contra el árbol exacto), pero con
  P finito puede extinguirse antes que el árbol real: con P = 24 murió a
  profundidad 75 y con P = 500 llegó hasta 269. N(j) cerca de la extinción y
  Σ_j N(j) son, por tanto, probablemente *subestimaciones*.
- **Medición**: un solo portátil (Windows/MinGW), con muchos procesos a la vez;
  tiempos orientativos. `cake_lpr` (verificador LRAT verificado formalmente) no
  se probó porque su binario apunta a Linux; habría que ejecutarlo en Linux/WSL
  sobre las mismas pruebas `.lrat`.
