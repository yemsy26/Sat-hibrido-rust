"""El motor desde Python, sin dependencias: ctypes sobre motor_ffi.

    cargo build --release -p motor-ffi
    python crates/motor-ffi/examples/ejemplo.py [--certify]

Parar desde otro hilo, con motor_stop, es seguro: ctypes suelta el GIL durante motor_solve.
"""
import ctypes
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
FILE = {"win32": "motor_ffi.dll", "darwin": "libmotor_ffi.dylib"}.get(sys.platform, "libmotor_ffi.so")
lib = ctypes.CDLL(str(ROOT / "target" / "release" / FILE))

lib.motor_new.restype = ctypes.c_void_p
lib.motor_free.argtypes = [ctypes.c_void_p]
lib.motor_add_clause.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
lib.motor_load_opb.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.motor_set_option.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lib.motor_solve.argtypes = [ctypes.c_void_p]
lib.motor_result_json.argtypes = [ctypes.c_void_p]
lib.motor_result_json.restype = ctypes.c_char_p


def result(solver):
    return json.loads(lib.motor_result_json(solver))


solver = lib.motor_new()
lib.motor_set_option(solver, b"certify", b"1" if "--certify" in sys.argv else b"0")
for clause in ([1, 2], [-1]):
    lib.motor_add_clause(solver, (ctypes.c_int32 * len(clause))(*clause), len(clause))
code = lib.motor_solve(solver)
print("sat:", code, result(solver)["model"])
assert code == 10 and result(solver)["model"] == [-1, 2]

lib.motor_add_clause(solver, (ctypes.c_int32 * 1)(-2), 1)
code = lib.motor_solve(solver)
print("unsat:", code, "certificado:", result(solver)["proof_checked"])
assert code == 20 and result(solver)["proof_checked"] == ("--certify" in sys.argv)

lib.motor_load_opb(solver, b"min: +3 x1 +2 x2 +4 x3 ;\n+1 x1 +1 x2 +1 x3 >= 2 ;\n")
code = lib.motor_solve(solver)
print("optimo:", code, "coste:", result(solver)["cost"])
assert code == 30 and result(solver)["cost"] == 5
lib.motor_free(solver)
