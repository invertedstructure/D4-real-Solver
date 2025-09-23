
import numpy as np
from .gf2 import to_bool, matmul_gf2, eq_gf2, in_image

class ChainComplex:
    def __init__(self, boundaries):
        self.boundaries = {int(k): to_bool(v) for k, v in boundaries.items()}
        self.maxdeg = max(self.boundaries.keys()) if self.boundaries else -1
        self.dims = {}
        for k, d in self.boundaries.items():
            self.dims[k] = d.shape[1]
            self.dims[k-1] = d.shape[0]

    def d(self, k):
        return self.boundaries.get(k, np.zeros((self.dims.get(k-1,0), self.dims.get(k,0)), dtype=bool))

def check_boundary_compat(CX, CY, Cmap, zlift=False, dX_signed=None, dY_signed=None, C_signed=None):
    ok = True; details = []
    if not zlift:
        degs = sorted(Cmap.keys())
        for k in degs:
            if (k-1) not in Cmap:
                continue
            Ck = to_bool(Cmap[k])
            Ckm1 = to_bool(Cmap[k-1])
            dY = CY.d(k); dX = CX.d(k)
            lhs = matmul_gf2(dY, Ck)
            rhs = matmul_gf2(Ckm1, dX)
            eq = eq_gf2(lhs, rhs)
            ok &= eq; details.append((int(k), bool(eq)))
        return ok, details
    else:
        degs = sorted(C_signed.keys())
        for k in degs:
            if (k-1) not in C_signed:
                continue
            Ck = C_signed[k]; Ckm1 = C_signed[k-1]
            dY = dY_signed.get(k); dX = dX_signed.get(k)
            lhs = dY @ Ck; rhs = Ckm1 @ dX
            eq = (lhs.shape == rhs.shape) and (lhs == rhs).all()
            ok &= eq; details.append((int(k), bool(eq)))
        return ok, details

def check_transport_homology(CX, CY, Cmap, c_dom, c_cod, k):
    v_map = (to_bool(Cmap[k]) @ to_bool(c_dom).reshape(-1,1)) % 2
    v_cod = to_bool(c_cod).reshape(-1,1)
    diff = (v_map.astype(int) ^ v_cod.astype(int)) % 2
    B = CY.d(k+1)
    # Safe handling at top degree
    if B.shape[0] == 0:
        in_im = np.all(diff == 0)
        return in_im, v_map, v_cod, diff
    try:
        in_im = in_image(B, diff)
    except ValueError as e:
        # Row mismatch indicates inconsistent dims; re-raise with more context
        raise ValueError(f"transport check at degree {k}: rows(dY_{k+1})={B.shape[0]}, len(diff)={diff.shape[0]} :: {e}")
    return in_im, v_map, v_cod, diff

def pairing_value(v_hi, v_lo, B=None, zlift=False, B_signed=None):
    if not zlift:
        v_hi = to_bool(v_hi).reshape(-1,1); v_lo = to_bool(v_lo).reshape(-1,1)
        if B is not None:
            B = to_bool(B); return int((v_hi.T @ (B.astype(int)) @ v_lo) % 2)
        n = min(v_hi.shape[0], v_lo.shape[0]); return int((v_hi[:n].T @ v_lo[:n]) % 2)
    else:
        v_hi = np.array(v_hi, dtype=int).reshape(-1,1); v_lo = np.array(v_lo, dtype=int).reshape(-1,1)
        if B_signed is not None: return int((v_hi.T @ B_signed @ v_lo).item())
        n = min(v_hi.shape[0], v_lo.shape[0]); return int((v_hi[:n].T @ v_lo[:n]).item())

def check_support(Cmap, support_idx):
    for k, M in Cmap.items():
        M = to_bool(M)
        rows = set(support_idx.get(str(k), {}).get("rows", [])) | set(support_idx.get(k, {}).get("rows", []))
        cols = set(support_idx.get(str(k), {}).get("cols", [])) | set(support_idx.get(k, {}).get("cols", []))
        nz = np.argwhere(M)
        for r,c in nz:
            if rows and r not in rows: return False
            if cols and c not in cols: return False
    return True
