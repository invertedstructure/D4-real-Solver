import numpy as np
from .gf2 import to_bool, matmul_gf2, add_gf2, eq_gf2, in_image
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
        for k, Ck in Cmap.items():
            Ck = to_bool(Ck)
            dY = CY.d(k); dX = CX.d(k)
            lhs = matmul_gf2(dY, Ck); rhs = matmul_gf2(Ck, dX)
            eq = eq_gf2(lhs, rhs); ok &= eq; details.append((k, eq))
        return ok, details
    else:
        for k, Ck in C_signed.items():
            dY = dY_signed.get(k); dX = dX_signed.get(k)
            lhs = dY @ Ck; rhs = Ck @ dX
            eq = (lhs.shape == rhs.shape) and (lhs == rhs).all(); ok &= eq; details.append((k, eq))
        return ok, details
def check_transport_homology(CX, CY, Cmap, c_dom, c_cod, k):
    v_map = (to_bool(Cmap[k]) @ to_bool(c_dom).reshape(-1,1)) % 2
    v_cod = to_bool(c_cod).reshape(-1,1)
    diff = (v_map.astype(int) ^ v_cod.astype(int)) % 2
    dYkp1 = CY.d(k+1); from .gf2 import in_image
    in_im = in_image(dYkp1, diff); return in_im, v_map, v_cod, diff
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
    from .gf2 import to_bool
    for k, M in Cmap.items():
        M = to_bool(M)
        rows = set(support_idx.get(str(k), {}).get("rows", [])) | set(support_idx.get(k, {}).get("rows", []))
        cols = set(support_idx.get(str(k), {}).get("cols", [])) | set(support_idx.get(k, {}).get("cols", []))
        nz = np.argwhere(M)
        for r,c in nz:
            if rows and r not in rows: return False
            if cols and c not in cols: return False
    return True
