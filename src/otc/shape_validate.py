def _shape_of(mat):
    rows = len(mat)
    cols = 0 if rows == 0 else len(mat[0])
    for r in mat:
        if len(r) != cols:
            raise ValueError("Ragged matrix row lengths.")
    return rows, cols

def enforce_chain_shapes(complex_json, manifest):
    for k_str, d_k in complex_json["boundaries"].items():
        rows, cols = _shape_of(d_k)
        want_rows = manifest["degrees"][k_str]["dim_k_minus_1"]
        want_cols = manifest["degrees"][k_str]["dim_k"]
        if rows != want_rows or cols != want_cols:
            raise ValueError(f"d_{k_str} shape {rows}x{cols} != {want_rows}x{want_cols}")

def enforce_map_shapes(map_json, manifest, side="X"):
    for k_str, Ck in map_json["blocks"].items():
        rows, cols = _shape_of(Ck)
        want_cols = manifest["degrees"][k_str]["dim_k"]
        if cols != want_cols:
            raise ValueError(f"C_{k_str} cols {cols} != dim_k({side}) {want_cols}")

def enforce_rep_lengths(reps_json, manifest):
    k3 = str(int(reps_json["k3"])); k2 = str(int(reps_json["k2"]))
    if len(reps_json["c3_dom"]) != manifest["degrees"][k3]["dim_k"]:
        raise ValueError("c3_dom length mismatch")
    if len(reps_json["c3_cod"]) != manifest["degrees"][k3]["dim_k"]:
        raise ValueError("c3_cod length mismatch")
    if len(reps_json["c2_dom"]) != manifest["degrees"][k2]["dim_k"]:
        raise ValueError("c2_dom length mismatch")
    if len(reps_json["c2_cod"]) != manifest["degrees"][k2]["dim_k"]:
        raise ValueError("c2_cod length mismatch")

def enforce_support_bounds(support_json, manifest):
    for k_str, spec in support_json.items():
        max_row = manifest["degrees"][k_str]["dim_k_minus_1"]
        max_col = manifest["degrees"][k_str]["dim_k"]
        for r in spec.get("rows", []):
            if r < 0 or r >= max_row:
                raise ValueError(f"row index {r} out of bounds for degree {k_str}")
        for c in spec.get("cols", []):
            if c < 0 or c >= max_col:
                raise ValueError(f"col index {c} out of bounds for degree {k_str}")
