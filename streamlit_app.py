
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st, json, numpy as np, pandas as pd
from io import BytesIO

from otc.app_helpers import (
    load_complex, load_map_blocks, load_signed_blocks,
    unit_test_generator, overlap_test, triangle_test, run_tower
)
from otc.shape_validate import (
    enforce_chain_shapes, enforce_map_shapes, enforce_rep_lengths, enforce_support_bounds
)
from otc.triangle_builder import build_triangle_template

st.set_page_config(page_title="OTC 4D Sanity Runner (v3.4)", layout="wide")
st.title("Odd-Tetra Certificate — 4D Sanity Runner (v3.4)")
st.caption("Triangle builder: derive A,B,J from two moves. Keeps all previous features.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unit tests", "Overlaps & Triangle", "Towers & Novelty", "Runbook", "Notes"])

with tab1:
    st.header("Unit tests (per generator)")
    zlift = st.checkbox("Z-lift (signed boundary & pairing checks)", value=False)
    col1, col2 = st.columns(2)
    with col1:
        cx = st.file_uploader("Domain complex X (JSON)", type=["json"], key="u_cx")
        cy = st.file_uploader("Codomain complex Y (JSON)", type=["json"], key="u_cy")
        cmap = st.file_uploader("Move blocks C(m): C(X)->C(Y) (JSON)", type=["json"], key="u_cmap")
        support = st.file_uploader("Support indices (optional JSON)", type=["json"], key="u_supp")
        shapes = st.file_uploader("Shape manifest (JSON)", type=["json"], key="u_shapes")
    with col2:
        reps = st.file_uploader("Representatives & degrees (JSON)", type=["json"], key="u_reps")
        pairing = st.file_uploader("Pairing matrix (optional, GF2 JSON)", type=["json"], key="u_pair")
        dX = dY = C_signed = B_signed = None
        if zlift:
            dX = st.file_uploader("Signed d_k(X) blocks (JSON)", type=["json"], key="u_dX")
            dY = st.file_uploader("Signed d_k(Y) blocks (JSON)", type=["json"], key="u_dY")
            Csig = st.file_uploader("Signed C(m)_k blocks (JSON)", type=["json"], key="u_Csig")
            Bsig = st.file_uploader("Signed pairing matrix (optional JSON)", type=["json"], key="u_Bsig")

    unit_result = None
    if st.button("Run unit checks", type="primary"):
        try:
            assert cx and cy and cmap and reps, "Upload X, Y, Cmap, reps"
            CXj = json.load(cx); CYj = json.load(cy)
            Cmapj = json.load(cmap); repsd = json.load(reps)
            supp = json.load(support) if support else None
            shapesj = json.load(shapes) if shapes else None
            if shapesj:
                enforce_chain_shapes(CXj, shapesj); enforce_chain_shapes(CYj, shapesj)
                enforce_map_shapes(Cmapj, shapesj, side="X"); enforce_rep_lengths(repsd, shapesj)
                if supp: enforce_support_bounds(supp, shapesj)
            CX = load_complex(CXj); CY = load_complex(CYj)
            Cmapb = load_map_blocks(Cmapj)
            pair = json.load(pairing) if pairing else None
            dX_signed = load_signed_blocks(json.load(dX)) if (zlift and dX) else None
            dY_signed = load_signed_blocks(json.load(dY)) if (zlift and dY) else None
            C_signed = load_signed_blocks(json.load(Csig)) if (zlift and Csig) else None
            B_signed = np.array(json.load(Bsig), dtype=int) if (zlift and Bsig) else None
            unit_result = unit_test_generator(CX, CY, Cmapb, repsd, pair, supp, zlift, dX_signed, dY_signed, C_signed, B_signed)
            st.success("Unit checks completed."); st.json(unit_result)
        except Exception as e:
            st.error(f"Validation or run error: {e}")

with tab2:
    st.header("Overlaps & Triangle")
    sub = st.radio("Choose", ["Overlap (pairwise)", "Triangle coherence", "Build Triangle Template"], horizontal=True)
    if sub == "Overlap (pairwise)":
        c_overlap = st.file_uploader("Overlap complex (JSON)", type=["json"], key="ov_c")
        cm1 = st.file_uploader("Blocks C(m1) (JSON)", type=["json"], key="ov_m1")
        cm2 = st.file_uploader("Blocks C(m2) (JSON)", type=["json"], key="ov_m2")
        H = st.file_uploader("Homotopy H blocks (JSON)", type=["json"], key="ov_H")
        shapes = st.file_uploader("Shape manifest (JSON)", type=["json"], key="ov_shapes")
        if st.button("Run overlap test"):
            try:
                assert c_overlap and cm1 and cm2 and H, "Upload overlap complex, C(m1), C(m2), H"
                COj = json.load(c_overlap); C1j = json.load(cm1); C2j = json.load(cm2); Hj = json.load(H)
                shapesj = json.load(shapes) if shapes else None
                if shapesj:
                    enforce_chain_shapes(COj, shapesj)
                    enforce_map_shapes(C1j, shapesj, side="X"); enforce_map_shapes(C2j, shapesj, side="X")
                    enforce_map_shapes(Hj, shapesj, side="X")
                CO = load_complex(COj); C1 = load_map_blocks(C1j); C2 = load_map_blocks(C2j); Hb = load_map_blocks(Hj)
                ok, res = overlap_test(CO, C1, C2, Hb)
                st.write("PASS" if ok else "FAIL"); st.json(res)
            except Exception as e:
                st.error(f"Validation or run error: {e}")
    elif sub == "Triangle coherence":
        c_overlap = st.file_uploader("Overlap complex (JSON)", type=["json"], key="tri_c")
        Jfile = st.file_uploader("Triangle template J (JSON)", type=["json"], key="tri_J")
        shapes = st.file_uploader("Shape manifest (JSON)", type=["json"], key="tri_shapes")
        if st.button("Run triangle coherence test"):
            try:
                assert c_overlap and Jfile, "Upload overlap complex and J template"
                COj = json.load(c_overlap); Jj = json.load(Jfile); shapesj = json.load(shapes) if shapes else None
                if shapesj:
                    enforce_chain_shapes(COj, shapesj)
                    for k, part in Jj.items():
                        for key in ("A","B","J"):
                            if key in part:
                                enforce_map_shapes({"blocks": {k: part[key]}}, shapesj, side="X")
                CO = load_complex(COj)
                ok, res = triangle_test(CO, Jj)
                st.write("PASS" if ok else "FAIL"); st.json(res)
            except Exception as e:
                st.error(f"Validation or run error: {e}")
    else:
        st.subheader("Build Triangle Template from two moves")
        cx = st.file_uploader("Complex X (JSON)", type=["json"], key="tb_cx")
        cm1 = st.file_uploader("Blocks C(m1) (JSON)", type=["json"], key="tb_m1")
        cm2 = st.file_uploader("Blocks C(m2) (JSON)", type=["json"], key="tb_m2")
        shapes = st.file_uploader("Shape manifest (JSON)", type=["json"], key="tb_shapes")
        if st.button("Build template"):
            try:
                assert cx and cm1 and cm2, "Upload X, C(m1), C(m2)"
                CXj = json.load(cx); C1j = json.load(cm1); C2j = json.load(cm2)
                shapesj = json.load(shapes) if shapes else None
                if shapesj:
                    enforce_chain_shapes(CXj, shapesj); enforce_map_shapes(C1j, shapesj, side="X"); enforce_map_shapes(C2j, shapesj, side="X")
                from otc.app_helpers import load_complex, load_map_blocks
                CX = load_complex(CXj); C1 = load_map_blocks(C1j); C2 = load_map_blocks(C2j)
                Jbuilt = build_triangle_template(CX, C1, C2)
                st.json(Jbuilt)
                import json as _json
                bytes_out = _json.dumps(Jbuilt, indent=2).encode("utf-8")
                st.download_button("Download triangle_J_built.json", bytes_out, "triangle_J_built.json", "application/json")
            except Exception as e:
                st.error(f"Builder error: {e}")

with tab3:
    st.header("Towers & Novelty (GF2)")
    st.markdown("Upload schedule moves and a manifest; get per-step hashes and CSV export.")
    reps = st.file_uploader("Representatives & degrees (JSON)", type=["json"], key="tw_reps")
    shapes = st.file_uploader("Shape manifest (JSON)", type=["json"], key="tw_shapes")
    num = st.number_input("Number of steps in schedule", min_value=1, max_value=200, value=5, step=1)
    move_files = st.file_uploader("Upload move blocks for each step (JSON, in order)", type=["json"], accept_multiple_files=True, key="tw_moves")
    novelty_step = st.number_input("Novelty step (optional; 0 = none)", min_value=0, max_value=200, value=0, step=1)
    novelty_map = st.file_uploader("Novelty move blocks (JSON)", type=["json"], key="tw_nov")
    if st.button("Run tower"):
        try:
            assert reps and move_files and len(move_files) >= num, "Upload reps and moves"
            repsd = json.load(reps); shapesj = json.load(shapes) if shapes else None
            seq = []
            for f in move_files[:num]:
                Cj = json.load(f)
                if shapesj: enforce_map_shapes(Cj, shapesj, side="X")
                from otc.app_helpers import load_map_blocks
                seq.append(load_map_blocks(Cj))
            from otc.app_helpers import run_tower
            base_hashes = run_tower(None, seq, repsd)
            df = pd.DataFrame(base_hashes)
            st.subheader("Baseline tower hashes"); st.dataframe(df)
            st.download_button("Download tower-hashes.csv", df.to_csv(index=False).encode("utf-8"),
                               "tower-hashes.csv", "text/csv")
            if novelty_step > 0:
                assert novelty_map, "Upload novelty map"
                novj = json.load(novelty_map)
                if shapesj: enforce_map_shapes(novj, shapesj, side="X")
                from otc.app_helpers import load_map_blocks
                nov = load_map_blocks(novj)
                seq_nov = [(nov if i+1 == novelty_step else C) for i, C in enumerate(seq)]
                dfn = pd.DataFrame(run_tower(None, seq_nov, repsd))
                st.subheader("Tower with novelty injection"); st.dataframe(dfn)
                st.download_button("Download tower-novelty-hashes.csv", dfn.to_csv(index=False).encode("utf-8"),
                                   "tower-novelty-hashes.csv", "text/csv")
                div = None
                for i in range(min(len(df), len(dfn))):
                    if df.loc[i, "hash"] != dfn.loc[i, "hash"]:
                        div = i+1; break
                if div: st.error(f"Novelty detected at step {div}.")
                else: st.info("No divergence detected in hashes.")
        except Exception as e:
            st.error(f"Validation or run error: {e}")

with tab4:
    st.header("Runbook")
    st.markdown("Same as previous versions — download JSON and export all CSVs from here.")
    if "runbook" in st.session_state and len(st.session_state.runbook)>0:
        df = pd.DataFrame(st.session_state.runbook); st.dataframe(df, use_container_width=True)

with tab5:
    st.markdown("Notes: v3.4 adds a Triangle Template Builder from two maps (commutator-based).")
