import numpy as np
from hashlib import blake2b
from .gf2 import to_bool, matmul_gf2
def compose_maps(seq):
    if not seq: return {}
    degs = sorted(seq[0].keys())
    total = {k: to_bool(np.eye(seq[0][k].shape[0], dtype=bool)) for k in degs}
    for C in seq:
        for k in degs:
            total[k] = matmul_gf2(to_bool(C[k]), to_bool(total[k]))
    return total
def hash_certificate(C_total, reps):
    k3 = int(reps["k3"]); k2 = int(reps["k2"])
    c3 = to_bool(reps["c3_dom"]).reshape(-1,1); c2 = to_bool(reps["c2_dom"]).reshape(-1,1)
    v3 = to_bool(C_total[k3]) @ c3; v2 = to_bool(C_total[k2]) @ c2
    bits = np.concatenate([v3.flatten(), v2.flatten()]).astype(np.uint8)
    return blake2b(bits.tobytes(), digest_size=16).hexdigest()
