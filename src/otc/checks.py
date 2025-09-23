
import numpy as np
from .gf2 import to_bool

def commutator_identity(CX, C_m1, C_m2, H):
    """Check d_{k+1} H_k + H_{k-1} d_k = C_m2(k) C_m1(k) + C_m1(k) C_m2(k) (GF(2)).
    Shapes:
      d_k:    (n_{k-1} x n_k)
      H_k:    (n_{k+1} x n_k)
      C_m*(k):(n_k x n_k)
    We only evaluate degrees where all required blocks are present.
    """
    ok = True; results = {}
    degs = sorted(set(list(C_m1.keys()) + list(C_m2.keys())))
    for k in degs:
        n_k = CX.dims.get(k, 0)
        n_km1 = CX.dims.get(k-1, 0)
        n_kp1 = CX.dims.get(k+1, 0)
        d_k   = CX.d(k)
        d_kp1 = CX.d(k+1)
        C1k = to_bool(C_m1.get(k, np.zeros((n_k, n_k), dtype=bool)))
        C2k = to_bool(C_m2.get(k, np.zeros((n_k, n_k), dtype=bool)))
        Hk  = to_bool(H.get(k,  np.zeros((n_kp1, n_k), dtype=bool)))
        Hkm1= to_bool(H.get(k-1,np.zeros((n_k,   n_km1), dtype=bool)))
        lhs = (d_kp1.astype(int) @ Hk.astype(int)) % 2
        rhs_l = (Hkm1.astype(int) @ d_k.astype(int)) % 2
        lhs = (lhs ^ rhs_l) % 2
        rhs = ((C2k.astype(int) @ C1k.astype(int)) ^ (C1k.astype(int) @ C2k.astype(int))) % 2
        eq = (lhs.shape == rhs.shape) and np.array_equal(lhs, rhs)
        ok &= eq
        results[int(k)] = dict(eq=bool(eq), n_k=int(n_k))
    return ok, results

def triangle_coherence_identity(CX, J):
    ok = True; results = {}
    for k_str, data in J.items():
        k = int(k_str)
        n_k = CX.dims.get(k, 0)
        n_kp1 = CX.dims.get(k+1, 0)
        d_k   = CX.d(k)
        d_kp1 = CX.d(k+1)
        A = to_bool(data.get("A", np.zeros((n_k, n_k), dtype=bool)))
        B = to_bool(data.get("B", np.zeros((n_k, n_k), dtype=bool)))
        Jk= to_bool(data.get("J", np.zeros((n_kp1, n_k), dtype=bool)))
        lhs = ((d_kp1.astype(int) @ Jk.astype(int)) ^ (Jk.astype(int) @ d_k.astype(int))) % 2
        rhs = (A.astype(int) ^ B.astype(int)) % 2
        eq = (lhs.shape == rhs.shape) and np.array_equal(lhs, rhs)
        ok &= eq; results[k] = dict(eq=bool(eq), n_k=int(n_k))
    return ok, results
