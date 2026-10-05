#!/usr/bin/env bash
# Fase 17: calibracion del motor hibrido sobre formulas de Ramsey.
#
#   ./calibrate_ramsey.sh [directorio_de_trabajo] > results/phase17-ramsey-calibration.csv
#
# Cada fila es una ejecucion de `hunt` con limite de tiempo (TIMEOUT, 120 s por defecto).
# Los veredictos de VeriPB sobre las trazas van a la salida de error. N = 43 no se toca.
set -u
cd "$(dirname "$0")"
DIR=${1:-ramsey_work}
LIMIT=${TIMEOUT:-120}
mkdir -p "$DIR"
cargo build --release --bin gen_ramsey --bin hunt >&2 || exit 1
GEN=target/release/gen_ramsey
HUNT=target/release/hunt

# gen <etiqueta> <S> <T> <N>
gen() { [ -f "$DIR/$1.cnf" ] || "$GEN" "$4" "$DIR/$1.cnf" --cliques "$2" "$3" 2>/dev/null; }
# run <etiqueta> [opciones de hunt]; FILE=<formula> si la etiqueta no es el nombre del fichero
run() { local label=$1; shift; "$HUNT" "$DIR/${FILE:-$label}.cnf" --timeout "$LIMIT" --csv "$label" "$@" 2>/dev/null | grep -v '^s '; }
# certify <etiqueta> [opciones]: en serie con traza, y la traza por el verificador oficial
certify() {
    local label=$1; shift
    run "$label" --proof "$DIR/$label.pbp" "$@"
    if command -v veripb >/dev/null; then
        echo "veripb $label $*: $(veripb --cnf "$DIR/$label.cnf" "$DIR/$label.pbp" 2>&1 | grep -v '^Running' | head -1)" >&2
    fi
    rm -f "$DIR/$label.pbp"
}

"$HUNT" --header

# A. R(5,5) con N = 15, 20, 25: todas satisfacibles (R(5,5) >= 43).
for n in 15 20 25; do
    gen "r55_$n" 5 5 "$n"
    run "r55_$n"
    run "r55_$n" --no-symmetry
    run "r55_$n" --interval 0
    run "r55_$n" --no-symmetry --interval 0
    certify "r55_$n"
    for workers in 4 16; do
        run "r55_$n" --workers "$workers"
        run "r55_$n" --workers "$workers" --no-symmetry
    done
    run "r55_$n" --workers 16 --no-symmetry --interval 0
done

# B. La capa de simetria sola sobre R(5,5): que descubre y cuanto pesa su traza.
for n in 15 20 25 30 35 38 40 42; do
    gen "r55_$n" 5 5 "$n"
    run "r55_$n" --preprocess-only --no-parity --proof "$DIR/r55_$n.sym.pbp"
    rm -f "$DIR/r55_$n.sym.pbp"
done

# C. Numeros de Ramsey conocidos, justo en el valor (UNSAT) y uno por debajo (SAT).
for spec in "3 3 6" "3 4 9" "3 5 14" "4 4 17" "4 4 18"; do
    set -- $spec
    label="r$1$2_$3"
    gen "$label" "$1" "$2" "$3"
    run "$label"
    run "$label" --no-symmetry
    certify "$label"
    run "$label" --workers 16
    run "$label" --workers 16 --no-symmetry
done
# R(3,6) = 18: mismas 153 variables que R(4,4) con N = 18, y fuera de alcance.
gen r36_18 3 6 18
run r36_18
run r36_18 --workers 16

# D. Frontera satisfacible de R(5,5): hasta donde encuentra coloraciones que existen.
for n in 30 32 33 34 35 36; do
    gen "r55_$n" 5 5 "$n"
    run "r55_$n" --no-symmetry
    run "r55_$n"
    run "r55_$n" --workers 16 --no-symmetry
    run "r55_$n" --workers 16
done

# E. Oraculo y enjambre donde hay busqueda de verdad: R(5,5) con N = 35 (SAT) y
#    R(4,4) con N = 18 (UNSAT). El enjambre no es determinista: tres repeticiones.
LIMIT=60
for label in r55_35 r44_18; do
    for flips in 1 4 16 64 256; do
        run "$label" --flips "$flips"
    done
    run "$label" --interval 0
    for interval in 100 2000 10000; do
        run "$label" --interval "$interval"
    done
    for repeat in 1 2 3; do
        for workers in 2 4 8 16; do
            run "$label" --workers "$workers"
        done
        run "$label" --workers 16 --no-share
        run "$label" --workers 16 --interval 0
        for cubes in 1 16 64; do
            run "$label" --workers 16 --cubes "$cubes"
        done
    done
done

# F. Ritmo de la traza en una busqueda que no termina: N = 38, 40 y 42 durante 60 s.
LIMIT=60
for n in 38 40 42; do
    gen "r55_$n" 5 5 "$n"
    run "r55_$n" --proof "$DIR/r55_$n.rate.pbp"
    rm -f "$DIR/r55_$n.rate.pbp"
done

# G. Mas presupuesto para el oraculo en la frontera: no la mueve.
LIMIT=170
for n in 36 38 40 42; do
    for flips in 1024 16384; do
        run "r55_$n" --no-symmetry --flips "$flips"
    done
done

# H. Cola larga en R(4,4) con N = 18: semillas en serie y repeticiones del enjambre.
LIMIT=30
for flips in 16 64; do
    for seed in $(seq 1 20); do
        FILE=r44_18 run "r44_18_seed$seed" --flips "$flips" --seed "$seed"
    done
done
for options in "--cubes 4" "--cubes 4 --no-share" "--cubes 64" "--cubes 64 --no-share"; do
    for repeat in 1 2 3 4 5 6 7 8; do
        # shellcheck disable=SC2086
        FILE=r44_18 run r44_18_swarm --workers 16 $options
    done
done
