
import numpy as np
from .gf2 import to_bool

def commutator_identity(CX, C_m1, C_m2, H):
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
    """Check degreewise:  d_{k+1} J_k  ⊕  J_{k-1} d_k  =  A_k ⊕ B_k  over GF(2).
    Shapes:
      d_k:    (n_{k-1} x n_k)
      J_k:    (n_{k+1} x n_k)
      J_{k-1}:(n_k x n_{k-1})
      A_k,B_k:(n_k x n_k)
    We only evaluate degrees k that appear in J (for A_k,B_k) and where shapes are consistent.
    Missing J blocks default to zeros of the appropriate shape.
    """
    ok = True; results = {}
    # keys where we have A_k/B_k expectations
    for k_str, data in J.items():
        k = int(k_str)
        n_k = CX.dims.get(k, 0)
        n_km1 = CX.dims.get(k-1, 0)
        n_kp1 = CX.dims.get(k+1, 0)
        d_k   = CX.d(k)      # (n_{k-1} x n_k)
        d_kp1 = CX.d(k+1)    # (n_k x n_{k+1})
        # A,B,J_k provided at degree k
        A = to_bool(data.get("A", np.zeros((n_k, n_k), dtype=bool)))
        B = to_bool(data.get("B", np.zeros((n_k, n_k), dtype=bool)))
        Jk= to_bool(data.get("J", np.zeros((n_kp1, n_k), dtype=bool)))
        # J_{k-1} may or may not be present; fetch from J dict if exists, else zero of correct shape
        data_km1 = J.get(str(k-1), {}) if isinstance(J, dict) else {}
        Jkm1 = to_bool(data_km1.get("J", np.zeros((n_k, n_km1), dtype=bool)))
        # Compute
        term1 = (d_kp1.astype(int) @ Jk.astype(int)) % 2         # (n_k x n_k)
        term2 = (Jkm1.astype(int) @ d_k.astype(int)) % 2         # (n_k x n_k)
        lhs = (term1 ^ term2) % 2
        rhs = (A.astype(int) ^ B.astype(int)) % 2
        eq = (lhs.shape == rhs.shape) and np.array_equal(lhs, rhs)
        ok &= eq; results[k] = dict(eq=bool(eq), n_k=int(n_k))
    return ok, results
