# Comparacion con CaDiCaL y Kissat

Generado por `scripts/comparar_solvers.py report`.

Como leerlo:

* `motor` lleva `--no-verify` en las dos primeras tablas; el coste de certificar esta en la ultima.
* Muestra: 60 de las 279 instancias de hasta 1 MiB comprimido, de las 400 de la pista. Quedan fuera las mayores.
* Limite de 120 s en la competicion, donde el real es de 5000 s: favorece a quien responde pronto.
* Kissat rel-4.0.4 no admite Windows: se compilo con un parche minimo y `--quiet`, sin ajustar nada mas.
* Cuatro trabajos a la vez en 16 hilos con nucleos mixtos: los tiempos pequenos son orientativos.
* "Solo este" cuenta `motor` y `motor-4h` como un solo solver.
* Kissat no se certifico: la comparacion con prueba es solo contra CaDiCaL.

## SAT Competition 2023, pista main (limite de 120 s por instancia)

| Solver | Resueltas | SAT | UNSAT | Solo este | PAR-2 (s) |
|---|---:|---:|---:|---:|---:|
| kissat | 30/60 | 12 | 18 | 1 | 7798 |
| cadical | 21/60 | 13 | 8 | 4 | 9794 |
| motor | 27/60 | 8 | 19 | 2 | 8194 |
| motor-4h | 28/60 | 7 | 21 | 4 | 7974 |

| Instancia | kissat | cadical | motor | motor-4h |
|---|---:|---:|---:|---:|
| 008e7716c69009090ff6bd00 | UNSAT 17.2 | UNSAT 16.8 | UNSAT 34.4 | UNSAT 37.3 |
| 00f2eb377986e7decbc86393 | UNSAT 0.0 | UNSAT 0.2 | UNSAT 0.0 | UNSAT 0.0 |
| 02066c116dbacc40ec5cca20 | UNSAT 5.1 | UNSAT 3.6 | UNSAT 23.5 | UNSAT 17.3 |
| 037c423f56548082b1935e88 | SAT 0.3 | SAT 0.3 | SAT 0.2 | SAT 0.1 |
| 06b138e8a6da806e88ac7da2 | timeout | UNSAT 52.4 | timeout | timeout |
| 0ae0c5a4fb9001d63669a0aa | timeout | timeout | timeout | timeout |
| 1153a473e1d5dd21ccd33189 | timeout | timeout | timeout | timeout |
| 189e5991b041a89700513106 | timeout | timeout | timeout | timeout |
| 1a20649a8c79d6675d934cec | timeout | timeout | timeout | timeout |
| 1aa7cd96ab1f7bf58984decc | UNSAT 0.0 | UNSAT 1.4 | UNSAT 0.0 | UNSAT 0.0 |
| 1f215812371f0c428eb1f1e5 | timeout | timeout | timeout | timeout |
| 21bde0ebb111c637a1ae8789 | UNSAT 3.2 | timeout | UNSAT 0.1 | UNSAT 0.1 |
| 246afd75cb97a21144f368c0 | UNSAT 0.6 | timeout | UNSAT 0.0 | UNSAT 0.0 |
| 262009b8df6350761c7dfc87 | timeout | timeout | timeout | timeout |
| 29bab0de14e56e526be96831 | SAT 1.5 | SAT 4.8 | SAT 22.6 | timeout |
| 2d0c041c0fe72dc32527bfbf | SAT 1.4 | SAT 17.4 | SAT 47.0 | SAT 6.8 |
| 328da7966b09b2f6e99c93c4 | timeout | timeout | timeout | timeout |
| 37d40a1092b58ad28285b945 | UNSAT 6.9 | timeout | UNSAT 0.0 | UNSAT 0.0 |
| 396245861b331d45e1345bd4 | SAT 29.0 | SAT 36.4 | timeout | SAT 3.6 |
| 4029fbae284eaf924b37b6f4 | UNSAT 26.1 | timeout | UNSAT 0.1 | UNSAT 0.1 |
| 431bfb466be51ecd86b321b5 | UNSAT 0.3 | UNSAT 0.6 | UNSAT 0.7 | UNSAT 0.4 |
| 44b249f51321a2bf38f73053 | SAT 0.1 | SAT 0.1 | SAT 0.1 | SAT 0.1 |
| 4d2f74ef97b6d40ebd71f85d | timeout | timeout | UNSAT 0.0 | UNSAT 0.0 |
| 4fe42891b004d7425631a8ed | timeout | timeout | timeout | timeout |
| 558ce551d2b933b7a5f1c7d2 | UNSAT 115.6 | timeout | UNSAT 0.1 | UNSAT 0.1 |
| 587150fb7b12a6b5dd7e4a94 | SAT 0.1 | SAT 0.0 | SAT 0.0 | SAT 0.0 |
| 63159b6b10c3d6832bd5944e | timeout | SAT 13.6 | SAT 22.0 | timeout |
| 6aacae96461cb65176942b6d | timeout | timeout | timeout | timeout |
| 711ff7681711949f44b07a0d | timeout | timeout | timeout | UNSAT 92.6 |
| 71cf538aa0751d27f144de9c | timeout | timeout | timeout | timeout |
| 751afd7ddf8f242ed7bd9351 | timeout | timeout | timeout | timeout |
| 7aaf3275cbe217044ef305f0 | UNSAT 1.2 | timeout | UNSAT 0.1 | UNSAT 0.1 |
| 81d2158d506ea10497088374 | timeout | timeout | timeout | timeout |
| 835ac60ca83af6f334b64b61 | timeout | timeout | UNSAT 0.0 | UNSAT 0.0 |
| 8bcd3f3e269fa3c9516667b6 | SAT 49.2 | timeout | timeout | timeout |
| 8d458780fb74c28c01a5fe89 | UNSAT 3.6 | UNSAT 13.2 | UNSAT 40.3 | UNSAT 16.0 |
| 91b88d172a08e2a57c189bdd | timeout | timeout | timeout | timeout |
| 953e9ff735bdf5ada27b0ab2 | timeout | timeout | timeout | timeout |
| 96e01064ac3b644143278c21 | SAT 22.6 | timeout | SAT 10.6 | timeout |
| 9de6ddddf7fde06bf2efd955 | SAT 78.2 | SAT 5.6 | timeout | timeout |
| a18b6fbdfee43c007e07dedf | timeout | SAT 76.4 | timeout | timeout |
| a552a058e6376a36b1f1b272 | timeout | timeout | timeout | timeout |
| a8dc92a9c0434576b47c82c9 | timeout | timeout | timeout | timeout |
| aa4a861240c0c39b5aa44602 | timeout | timeout | timeout | timeout |
| af1e84bc2ab44d87d1c4c0cb | UNSAT 11.1 | timeout | UNSAT 0.0 | UNSAT 0.0 |
| b2145c28dbed385329ea73a0 | UNSAT 5.4 | timeout | UNSAT 0.1 | UNSAT 0.0 |
| b86f2425cafeb1239d65bd52 | timeout | timeout | timeout | timeout |
| c0069efa0c924e416878d942 | timeout | SAT 61.8 | timeout | timeout |
| c9af5b23c87350f5d817acc9 | UNSAT 43.4 | timeout | UNSAT 0.0 | UNSAT 0.0 |
| cd118ff851f95661b2a3adfc | timeout | timeout | timeout | SAT 62.2 |
| cfc53adcc8ed18ca161bb6f8 | UNSAT 93.0 | timeout | timeout | UNSAT 34.1 |
| d1f4348ad66cae01386777da | timeout | timeout | timeout | timeout |
| d84a89fb1bc1ff37dc473786 | SAT 74.8 | SAT 14.4 | timeout | timeout |
| dd169198070f9aa35015de65 | UNSAT 2.4 | timeout | UNSAT 0.0 | UNSAT 0.0 |
| e46bdd62b3789b64f7fbb3bd | timeout | timeout | timeout | timeout |
| ed76b075cdf07435c6e4f620 | timeout | SAT 50.8 | timeout | timeout |
| f08b3567642ea029452fff2c | timeout | timeout | timeout | timeout |
| f448874997bae64bd455fb00 | SAT 5.1 | timeout | SAT 71.4 | SAT 22.7 |
| fab0db1ab0dd5baa3215aab5 | SAT 0.9 | SAT 64.4 | timeout | timeout |
| fe19a31b76cbba5901e16ce3 | UNSAT 0.0 | UNSAT 0.0 | UNSAT 0.0 | UNSAT 0.0 |

## Familias generadas (limite de 300 s por instancia)

| Solver | Resueltas | SAT | UNSAT | Solo este | PAR-2 (s) |
|---|---:|---:|---:|---:|---:|
| kissat | 21/25 | 9 | 12 | 0 | 2540 |
| cadical | 19/25 | 9 | 10 | 0 | 3742 |
| motor | 25/25 | 9 | 16 | 4 | 105 |
| motor-4h | 25/25 | 9 | 16 | 4 | 100 |

| Instancia | kissat | cadical | motor | motor-4h |
|---|---:|---:|---:|---:|
| aleatorio_200_1 | UNSAT 0.5 | UNSAT 0.5 | UNSAT 0.2 | UNSAT 0.1 |
| aleatorio_200_2 | SAT 0.1 | SAT 0.2 | SAT 0.0 | SAT 0.0 |
| aleatorio_200_3 | SAT 0.2 | SAT 0.3 | SAT 0.1 | SAT 0.0 |
| aleatorio_250_1 | SAT 0.9 | SAT 2.4 | SAT 2.2 | SAT 0.0 |
| aleatorio_250_2 | UNSAT 2.3 | UNSAT 3.4 | UNSAT 3.0 | UNSAT 0.4 |
| aleatorio_250_3 | UNSAT 1.8 | UNSAT 2.0 | UNSAT 2.1 | UNSAT 0.2 |
| aleatorio_300_1 | UNSAT 32.0 | UNSAT 41.4 | UNSAT 72.4 | UNSAT 5.5 |
| aleatorio_300_2 | SAT 0.1 | SAT 0.5 | SAT 0.0 | SAT 0.0 |
| aleatorio_300_3 | SAT 1.0 | SAT 0.5 | SAT 0.5 | SAT 0.1 |
| palomar_10 | UNSAT 1.7 | UNSAT 61.3 | UNSAT 0.0 | UNSAT 0.0 |
| palomar_11 | UNSAT 70.2 | timeout | UNSAT 0.0 | UNSAT 0.0 |
| palomar_12 | timeout | timeout | UNSAT 0.0 | UNSAT 0.0 |
| palomar_9 | UNSAT 0.3 | UNSAT 3.4 | UNSAT 0.0 | UNSAT 0.0 |
| ramsey_3_5_k13 | SAT 0.0 | SAT 0.0 | SAT 0.3 | SAT 0.2 |
| ramsey_3_5_k14 | timeout | timeout | UNSAT 0.4 | UNSAT 0.3 |
| ramsey_3_6_k17 | SAT 0.0 | SAT 0.0 | SAT 0.8 | SAT 0.7 |
| ramsey_3_6_k18 | timeout | timeout | UNSAT 0.9 | UNSAT 0.8 |
| ramsey_4_4_k17 | SAT 0.1 | SAT 0.1 | SAT 1.1 | SAT 0.6 |
| ramsey_4_4_k18 | timeout | timeout | UNSAT 1.1 | UNSAT 0.9 |
| ramsey_4_5_k24 | SAT 2.1 | SAT 24.4 | SAT 19.6 | SAT 90.4 |
| tseitin_30 | UNSAT 0.0 | UNSAT 0.0 | UNSAT 0.0 | UNSAT 0.0 |
| tseitin_40 | UNSAT 0.0 | UNSAT 0.1 | UNSAT 0.0 | UNSAT 0.0 |
| tseitin_50 | UNSAT 0.0 | UNSAT 0.1 | UNSAT 0.0 | UNSAT 0.0 |
| tseitin_60 | UNSAT 0.1 | UNSAT 1.0 | UNSAT 0.0 | UNSAT 0.0 |
| tseitin_80 | UNSAT 26.2 | timeout | UNSAT 0.0 | UNSAT 0.0 |

## Con certificado

Tiempo total de resolver y comprobar la prueba, en los UNSAT que `motor` resolvio.

| Solver | Resueltas | SAT | UNSAT | Solo este | PAR-2 (s) |
|---|---:|---:|---:|---:|---:|
| motor+veripb | 35/35 | 0 | 35 | 18 | 1315 |
| cadical+drat-trim | 17/35 | 0 | 17 | 0 | 21925 |

| Instancia | motor+veripb | cadical+drat-trim |
|---|---:|---:|
| 008e7716c69009090ff6bd00 | UNSAT 446.3 | UNSAT 31.0 |
| 00f2eb377986e7decbc86393 | UNSAT 0.3 | UNSAT 0.4 |
| 02066c116dbacc40ec5cca20 | UNSAT 230.5 | UNSAT 6.2 |
| 1aa7cd96ab1f7bf58984decc | UNSAT 0.4 | UNSAT 3.1 |
| 21bde0ebb111c637a1ae8789 | UNSAT 0.4 | timeout |
| 246afd75cb97a21144f368c0 | UNSAT 0.1 | timeout |
| 37d40a1092b58ad28285b945 | UNSAT 0.1 | timeout |
| 4029fbae284eaf924b37b6f4 | UNSAT 0.5 | timeout |
| 431bfb466be51ecd86b321b5 | UNSAT 1.6 | UNSAT 1.2 |
| 4d2f74ef97b6d40ebd71f85d | UNSAT 13.4 | timeout |
| 558ce551d2b933b7a5f1c7d2 | UNSAT 0.8 | timeout |
| 7aaf3275cbe217044ef305f0 | UNSAT 0.2 | timeout |
| 835ac60ca83af6f334b64b61 | UNSAT 0.4 | timeout |
| 8d458780fb74c28c01a5fe89 | UNSAT 282.5 | UNSAT 29.7 |
| af1e84bc2ab44d87d1c4c0cb | UNSAT 0.2 | timeout |
| aleatorio_200_1 | UNSAT 0.8 | UNSAT 1.0 |
| aleatorio_250_2 | UNSAT 8.6 | UNSAT 5.9 |
| aleatorio_250_3 | UNSAT 6.1 | UNSAT 3.6 |
| aleatorio_300_1 | UNSAT 308.6 | UNSAT 63.2 |
| b2145c28dbed385329ea73a0 | UNSAT 0.2 | timeout |
| c9af5b23c87350f5d817acc9 | UNSAT 0.1 | timeout |
| dd169198070f9aa35015de65 | UNSAT 0.1 | timeout |
| fe19a31b76cbba5901e16ce3 | UNSAT 0.6 | UNSAT 0.1 |
| palomar_10 | UNSAT 0.1 | UNSAT 165.1 |
| palomar_11 | UNSAT 0.1 | timeout |
| palomar_12 | UNSAT 0.1 | timeout |
| palomar_9 | UNSAT 0.1 | UNSAT 10.3 |
| ramsey_3_5_k14 | UNSAT 1.9 | timeout |
| ramsey_3_6_k18 | UNSAT 1.6 | timeout |
| ramsey_4_4_k18 | UNSAT 7.8 | timeout |
| tseitin_30 | UNSAT 0.1 | UNSAT 0.1 |
| tseitin_40 | UNSAT 0.1 | UNSAT 0.3 |
| tseitin_50 | UNSAT 0.1 | UNSAT 0.2 |
| tseitin_60 | UNSAT 0.1 | UNSAT 3.8 |
| tseitin_80 | UNSAT 0.2 | timeout |

## Comprobaciones

410 ejecuciones. Cada modelo se evaluo contra la CNF, sin creer al solver.

Ningun modelo falso, ningun veredicto opuesto, ningun fallo de ejecucion.
