#!/usr/bin/env bash
# Fase 17: lanzamiento de R(5,5) con N = 43. La calibracion NO lo ejecuta.
#
#   ./run_ramsey_43.sh
#   OUT=/d/ramsey WORKERS=12 PROOF_CAP_GIB=200 ./run_ramsey_43.sh
#
# Dos procesos sobre la misma formula (903 variables, 1 925 196 clausulas):
#
#   enjambre  WORKERS hilos sin traza: el motor no certifica en paralelo. Es la via rapida
#             y la que daria una coloracion si existiera (R(5,5) >= 44).
#   notario   un hilo con traza VeriPB 3.0: la unica via a un UNSAT certificado. La traza
#             crece a unos 10 MB/s (36 GB por hora, medido en N = 42); al llegar a
#             PROOF_CAP_GIB deja de escribirse y el notario sigue SIN certificado.
#             NOTARY=0 lo suprime y da ese hilo al enjambre.
#
# Parametros deducidos de results/phase17-ramsey-calibration.csv:
#   FLIPS=64      con 16 el oraculo no resuelve N = 35 en la ignicion y con 64 si; en
#                 R(4,4) con N = 18 (UNSAT) 16 y 64 empatan. Sin oraculo nada termina.
#   INTERVAL=500  100 y 500 empatan; 2000 y 10000 son peores.
#   CUBES=64      1024 cubos, el tope del motor, con intercambio de clausulas: la unica
#                 configuracion sin ejecuciones atascadas en R(4,4) (11 de 11 en menos de
#                 1 s; el resto se atasca mas de 30 s entre el 35 y el 50 % de las veces).
#   simetria      activa: en N = 42 descubre un solo generador (40 predicados) en 1.6 s.
#
# Memoria: unos 0.4 GB por hilo (5.4 GB con 15 workers en N = 42). Progreso: enjambre.log y notario.log, una linea cada
# HEARTBEAT segundos; el motor no expone contadores mientras busca, asi que lo unico que
# se mueve es el tamano de la traza del notario.
#
# Codigo de salida: 10 coloracion encontrada, 20 UNSAT (certificado o no, segun run.log),
# 1 error o interrupcion.
set -u
cd "$(dirname "$0")"

# N y CLIQUES existen para ensayar el script con una formula pequena, p. ej.
#   N=18 CLIQUES="4 4" OUT=ensayo ./run_ramsey_43.sh
N=${N:-43}
CLIQUES=${CLIQUES:-5 5}
OUT=${OUT:-ramsey_$N}
NOTARY=${NOTARY:-1}
WORKERS=${WORKERS:-$(( $(nproc 2>/dev/null || echo 16) - NOTARY ))}
CUBES=${CUBES:-64}
FLIPS=${FLIPS:-64}
INTERVAL=${INTERVAL:-500}
PROOF_CAP_GIB=${PROOF_CAP_GIB:-64}
HEARTBEAT=${HEARTBEAT:-600}

mkdir -p "$OUT" || exit 1
cargo build --release --bin gen_ramsey --bin hunt || exit 1
CNF="$OUT/ramsey_$N.cnf"
# shellcheck disable=SC2086
[ -s "$CNF" ] || target/release/gen_ramsey "$N" "$CNF" --cliques $CLIQUES || exit 1
if [ "$N" = 43 ] && [ "$CLIQUES" = "5 5" ] && ! grep -q -m1 '^p cnf 903 1925196$' "$CNF"; then
    echo "El CNF no es R(5,5) con N = 43" >&2
    exit 1
fi
if [ "$NOTARY" = 1 ]; then
    free=$(df -Pk "$OUT" | awk 'NR == 2 { print int($4 / 1048576) }')
    if [ "$free" -lt $((PROOF_CAP_GIB + 16)) ]; then
        echo "Disco: $free GiB libres y la traza pide $PROOF_CAP_GIB mas 16 de margen. Baja PROOF_CAP_GIB o usa NOTARY=0." >&2
        exit 1
    fi
fi

say() { echo "$(date '+%F %T') $*" | tee -a "$OUT/run.log"; }
say "N=$N cliques=($CLIQUES) workers=$WORKERS cubos=$CUBES flips=$FLIPS intervalo=$INTERVAL notario=$NOTARY tope=${PROOF_CAP_GIB}GiB"

target/release/hunt "$CNF" --workers "$WORKERS" --cubes "$CUBES" --flips "$FLIPS" --interval "$INTERVAL" \
    --model "$OUT/enjambre.model" --heartbeat "$HEARTBEAT" --csv enjambre > "$OUT/enjambre.out" 2> "$OUT/enjambre.log" &
swarm=$!
notary=
if [ "$NOTARY" = 1 ]; then
    target/release/hunt "$CNF" --flips "$FLIPS" --interval "$INTERVAL" \
        --proof "$OUT/ramsey_$N.pbp" --proof-cap-gib "$PROOF_CAP_GIB" \
        --model "$OUT/notario.model" --heartbeat "$HEARTBEAT" --csv notario > "$OUT/notario.out" 2> "$OUT/notario.log" &
    notary=$!
fi
pending="$swarm $notary"
# shellcheck disable=SC2086
trap 'say "interrumpido"; kill $pending 2>/dev/null; exit 1' INT TERM

truncated() { grep -q TRUNCADA "$OUT/notario.log" 2>/dev/null; }
verdict=1
while [ -n "${pending// /}" ]; do
    # shellcheck disable=SC2086
    wait -n -p finished $pending
    code=$?
    who=enjambre
    [ "$finished" = "$notary" ] && who=notario
    pending=${pending/$finished/}
    case $code in
        10)
            say "$who: COLORACION ENCONTRADA y comprobada contra la formula: $OUT/$who.model"
            verdict=10
            break ;;
        20)
            verdict=20
            if [ "$who" = notario ] && ! truncated; then
                say "notario: UNSAT con traza completa en $OUT/ramsey_$N.pbp"
                if command -v veripb >/dev/null; then
                    say "veripb: $(veripb --cnf "$CNF" "$OUT/ramsey_$N.pbp" 2>&1 | grep -v '^Running' | head -1)"
                else
                    say "veripb no esta instalado: comprueba con veripb --cnf $CNF $OUT/ramsey_$N.pbp"
                fi
                break
            fi
            say "$who: UNSAT SIN CERTIFICADO"
            # Solo merece la pena esperar a un notario cuya traza sigue entera.
            if [ "$who" = enjambre ] && [ -n "${pending// /}" ] && ! truncated; then
                say "el notario sigue: su traza completa seria el certificado"
                continue
            fi
            break ;;
        *)
            say "$who termina sin veredicto (codigo $code): mira $OUT/$who.log" ;;
    esac
done
# shellcheck disable=SC2086
kill $pending 2>/dev/null
say "fin con codigo $verdict"
exit "$verdict"
