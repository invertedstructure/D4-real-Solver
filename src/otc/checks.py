import numpy as np
from .gf2 import to_bool
from .chain import ChainComplex
def commutator_identity(CX, C_m1, C_m2, H):
    ok = True; results = {}
    degs = sorted(set(list(C_m1.keys()) + list(C_m2.keys()) + list(H.keys())))
    for k in degs:
        d_k = CX.d(k)
        Hk = to_bool(H.get(k, np.zeros_like(d_k)))
        term = ((d_k.astype(int) @ Hk.astype(int)) ^
                (to_bool(H.get(k-1, np.zeros((Hk.shape[0], d_k.shape[0]), bool))).astype(int) @ d_k.astype(int))) % 2
        rhs = ((to_bool(C_m2.get(k, np.zeros_like(d_k))).astype(int) @ to_bool(C_m1.get(k, np.zeros_like(d_k))).astype(int)) ^
               (to_bool(C_m1.get(k, np.zeros_like(d_k))).astype(int) @ to_bool(C_m2.get(k, np.zeros_like(d_k))).astype(int))) % 2
        eq = (term.shape == rhs.shape) and np.array_equal(term, rhs)
        ok &= eq; results[int(k)] = dict(eq=bool(eq))
    return ok, results
def triangle_coherence_identity(CX, J):
    ok = True; results = {}
    for k, data in J.items():
        A = to_bool(data.get("A", [])); B = to_bool(data.get("B", [])); Jk = to_bool(data.get("J", []))
        d_k = CX.d(int(k))
        lhs = ((d_k.astype(int) @ Jk.astype(int)) ^ (Jk.astype(int) @ d_k.astype(int))) % 2
        rhs = (A.astype(int) ^ B.astype(int)) % 2
        eq = (lhs.shape == rhs.shape) and np.array_equal(lhs, rhs)
        ok &= eq; results[int(k)] = dict(eq=bool(eq))
    return ok, results
