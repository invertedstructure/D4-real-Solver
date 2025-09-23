import json, numpy as np, pandas as pd
from .chain import ChainComplex, check_boundary_compat, check_transport_homology, pairing_value, check_support
from .checks import commutator_identity, triangle_coherence_identity
from .towers import compose_maps, hash_certificate
def load_complex(json_obj):
    return ChainComplex(json_obj["boundaries"])
def load_map_blocks(json_obj):
    return {int(k): np.array(v, dtype=bool) for k, v in json_obj["blocks"].items()}
def load_signed_blocks(json_obj):
    return {int(k): np.array(v, dtype=int) for k, v in json_obj["blocks"].items()}
def unit_test_generator(CX, CY, Cmap, reps, pairing=None, support=None, zlift=False, dX_signed=None, dY_signed=None, C_signed=None, B_signed=None):
    boundary_ok, _ = check_boundary_compat(CX, CY, Cmap, zlift=zlift, dX_signed=dX_signed, dY_signed=dY_signed, C_signed=C_signed)
    c3_ok, *_ = check_transport_homology(CX, CY, Cmap, reps["c3_dom"], reps["c3_cod"], k=reps["k3"])
    c2_ok, *_ = check_transport_homology(CX, CY, Cmap, reps["c2_dom"], reps["c2_cod"], k=reps["k2"])
    pair_before = pairing_value(reps["c3_dom"], reps["c2_dom"], pairing, zlift=zlift, B_signed=B_signed)
    pair_after  = pairing_value(reps["c3_cod"], reps["c2_cod"], pairing, zlift=zlift, B_signed=B_signed)
    pair_ok = (pair_before == pair_after)
    support_ok = True
    if support is not None:
        support_ok = check_support(Cmap, support)
    return dict(boundary=boundary_ok, transport_c3=c3_ok, transport_c2=c2_ok, pairing=pair_ok, support=support_ok)
def overlap_test(C_overlap, C_m1, C_m2, H):
    ok, res = commutator_identity(C_overlap, C_m1, C_m2, H); return ok, res
def triangle_test(C_overlap, J):
    ok, res = triangle_coherence_identity(C_overlap, J); return ok, res
def run_tower(_, maps_seq, reps):
    hashes = []
    from .gf2 import to_bool, matmul_gf2
    degs = sorted(maps_seq[0].keys())
    cum = {k: to_bool(np.eye(maps_seq[0][k].shape[0], dtype=bool)) for k in degs}
    for i, C in enumerate(maps_seq, 1):
        for k in degs:
            cum[k] = matmul_gf2(to_bool(C[k]), to_bool(cum[k]))
        from .towers import hash_certificate
        h = hash_certificate(cum, reps)
        hashes.append(dict(step=i, hash=h))
    return hashes
