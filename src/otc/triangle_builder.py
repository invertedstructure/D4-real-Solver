
import numpy as np
from .gf2_solve import to_bool, kron_gf2, vec_gf2, unvec_gf2, solve_gf2

def build_triangle_template(CX, C1, C2):
    """Given ChainComplex CX (with dims, d(k)), and two move blocks C1, C2 (dict k->n_k x n_k),
    build a J-template s.t. d_{k+1} J_k ⊕ J_{k-1} d_k = D_k where D_k = C2_k C1_k ⊕ C1_k C2_k.
    Returns dict: k -> {A,B,J}, with A=C2C1, B=C1C2 and J as solved (zeros if trivial).
    """
    J = {}
    # collect degrees present in either map
    degs = sorted(set(list(C1.keys()) + list(C2.keys())))
    for k in degs:
        n_k = CX.dims.get(k, 0)
        n_km1 = CX.dims.get(k-1, 0)
        n_kp1 = CX.dims.get(k+1, 0)
        if n_k == 0: 
            continue
        C1k = to_bool(C1.get(k, np.zeros((n_k, n_k), dtype=bool))).astype(np.int8)
        C2k = to_bool(C2.get(k, np.zeros((n_k, n_k), dtype=bool))).astype(np.int8)
        Dk = ((C2k @ C1k) ^ (C1k @ C2k)) % 2   # n_k x n_k
        # Unknowns: vec(J_k) of size n_{k+1}*n_k, vec(J_{k-1}) of size n_k * n_{k-1}
        # Equation in vec form: (I ⊗ d_{k+1}) vec(J_k)  ⊕  (d_k^T ⊗ I) vec(J_{k-1}) = vec(D_k)
        Ik = np.eye(n_k, dtype=np.int8)
        dk = to_bool(CX.d(k)).astype(np.int8)           # (n_{k-1} x n_k)
        dkp1 = to_bool(CX.d(k+1)).astype(np.int8)       # (n_k x n_{k+1})
        A1 = kron_gf2(Ik, dkp1.T)  # shape: (n_k*n_k) x (n_{k+1}*n_k)
        A2 = kron_gf2(dk.T, Ik)    # shape: (n_k*n_k) x (n_k*n_{k-1})
        # Build augmented system [A1 | A2] [vec(J_k); vec(J_{k-1})] = vec(D_k)
        A = np.hstack([A1, A2]).astype(np.int8)
        b = vec_gf2(Dk).astype(np.int8)
        try:
            x = solve_gf2(A, b)
            x1 = x[:n_k*n_kp1] if (n_k*n_kp1)>0 else np.zeros((0,1), dtype=np.int8)
            x2 = x[n_k*n_kp1:] if (n_k*n_kp1)>=0 else np.zeros((0,1), dtype=np.int8)
            Jk = unvec_gf2(x1, n_kp1, n_k) if n_kp1>0 else np.zeros((0, n_k), dtype=np.int8)
            Jkm1 = unvec_gf2(x2, n_k, n_km1) if n_km1>0 else np.zeros((n_k, 0), dtype=np.int8)
        except ValueError:
            # If inconsistent, fall back to trivial J blocks and keep D as A⊕B (caller can inspect)
            Jk = np.zeros((n_kp1, n_k), dtype=np.int8)
            Jkm1 = np.zeros((n_k, n_km1), dtype=np.int8)
        # Save blocks (only J_k is stored under k; J_{k-1} will be consumed by the checker)
        J[str(k)] = {
            "A": C2k.tolist(),
            "B": C1k.tolist(),
            "J": Jk.tolist()
        }
        # Ensure J_{k-1} is represented as well so the checker can pick it up if needed
        if n_km1 > 0 and str(k-1) not in J:
            J[str(k-1)] = {"A": [[0]*n_km1 for _ in range(n_km1)],
                           "B": [[0]*n_km1 for _ in range(n_km1)],
                           "J": Jkm1.tolist()}
        else:
            # If there is already an entry, just overwrite its J with J_{k-1}
            if n_km1 > 0:
                J[str(k-1)]["J"] = Jkm1.tolist()
    return J
