#!/usr/bin/env bash
# Descarga (git clone) y compila las herramientas externas en sat/tools/ y
# los programas propios en sat/bin/. Probado con MinGW-w64 gcc 16 (Git Bash);
# en Linux basta quitar -static/-lpsapi si molestan.
# Versiones fijadas para que el resultado sea reproducible.
set -euo pipefail
cd "$(dirname "$0")"
CADICAL_TAG=rel-3.0.1        # c60730422e758ef1cebe7aeddf2dda31c996bf04
DRAT_TRIM_REV=2e3b2dc0ecf938addbd779d42877b6ed69d9a985
mkdir -p tools bin
[ -d tools/cadical ] || git clone -q --depth 1 --branch "$CADICAL_TAG" \
  https://github.com/arminbiere/cadical tools/cadical
if [ ! -d tools/drat-trim ]; then
  git clone -q https://github.com/marijnheule/drat-trim tools/drat-trim
  git -C tools/drat-trim checkout -q "$DRAT_TRIM_REV"
fi
EXTRA=""; [ "${OS:-}" = "Windows_NT" ] && EXTRA="-static -lpsapi"
(cd tools/cadical && CXX=g++ ./configure $EXTRA >/dev/null && make -C build -j8 cadical >/dev/null)
gcc -O2 -Dgetc_unlocked=getc -o tools/drat-trim/drat-trim tools/drat-trim/drat-trim.c
gcc -O2 -o tools/drat-trim/lrat-check tools/drat-trim/lrat-check.c
for p in enum_models cube_estimate; do
  g++ -O2 ${EXTRA%-lpsapi} -Itools/cadical/src -o bin/$p src/$p.cpp \
    tools/cadical/build/libcadical.a ${EXTRA#-static}
done
echo "listo: tools/cadical/build/cadical, tools/drat-trim/{drat-trim,lrat-check}, bin/*"
