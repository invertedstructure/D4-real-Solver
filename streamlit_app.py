
import sys, os
# Make local src/ importable on Streamlit Cloud
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st, json, numpy as np, pandas as pd
from otc.app_helpers import (
    load_complex, load_map_blocks, load_signed_blocks,
    unit_test_generator, overlap_test, triangle_test, run_tower
)

st.set_page_config(page_title="OTC 4D Sanity Runner (Ready Build)", layout="wide")
st.title("Odd-Tetra Certificate — 4D Sanity Runner (Ready Build)")

tab1, tab2, tab3, tab4 = st.tabs(["Unit tests", "Overlaps & Triangle", "Towers & Novelty", "Notes"])

with tab1:
    st.header("Unit tests (per generator)")
    zlift = st.checkbox("Z-lift (signed boundary & pairing checks)", value=False)
    col1, col2 = st.columns(2)
    with col1:
        cx = st.file_uploader("Domain complex X (JSON)", type=["json"], key="u_cx")
        cy = st.file_uploader("Codomain complex Y (JSON)", type=["json"], key="u_cy")
        cmap = st.file_uploader("Move blocks C(m): C(X)->C(Y) (JSON)", type=["json"], key="u_cmap")
        support = st.file_uploader("Support indices (optional JSON)", type=["json"], key="u_supp")
    with col2:
        reps = st.file_uploader("Representatives & degrees (JSON)", type=["json"], key="u_reps")
        pairing = st.file_uploader("Pairing matrix (optional, GF2 JSON)", type=["json"], key="u_pair")
        dX = dY = C_signed = B_signed = None
        if zlift:
            dX = st.file_uploader("Signed d_k(X) blocks (JSON)", type=["json"], key="u_dX")
            dY = st.file_uploader("Signed d_k(Y) blocks (JSON)", type=["json"], key="u_dY")
            Csig = st.file_uploader("Signed C(m)_k blocks (JSON)", type=["json"], key="u_Csig")
            Bsig = st.file_uploader("Signed pairing matrix (optional JSON)", type=["json"], key="u_Bsig")

    if st.button("Run unit checks", type="primary"):
        assert cx and cy and cmap and reps, "Upload X, Y, Cmap, reps"
        CX = load_complex(json.load(cx)); CY = load_complex(json.load(cy))
        Cmap = load_map_blocks(json.load(cmap))
        repsd = json.load(reps)
        pair = json.load(pairing) if pairing else None
        supp = json.load(support) if support else None
        dX_signed = load_signed_blocks(json.load(dX)) if (zlift and dX) else None
        dY_signed = load_signed_blocks(json.load(dY)) if (zlift and dY) else None
        C_signed = load_signed_blocks(json.load(Csig)) if (zlift and Csig) else None
        B_signed = np.array(json.load(Bsig), dtype=int) if (zlift and Bsig) else None
        res = unit_test_generator(CX, CY, Cmap, repsd, pair, supp, zlift, dX_signed, dY_signed, C_signed, B_signed)
        st.json(res)
        if zlift and (dX_signed is None or dY_signed is None or C_signed is None):
            st.warning("Z-lift enabled but signed blocks not fully provided. Boundary check may be incomplete.")

with tab2:
    st.header("Overlaps & Triangle")
    sub = st.radio("Choose", ["Overlap (pairwise)", "Triangle coherence"], horizontal=True)
    if sub == "Overlap (pairwise)":
        c_overlap = st.file_uploader("Overlap complex (JSON)", type=["json"], key="ov_c")
        cm1 = st.file_uploader("Blocks C(m1) (JSON)", type=["json"], key="ov_m1")
        cm2 = st.file_uploader("Blocks C(m2) (JSON)", type=["json"], key="ov_m2")
        H = st.file_uploader("Homotopy H blocks (JSON)", type=["json"], key="ov_H")
        if st.button("Run overlap test"):
            assert c_overlap and cm1 and cm2 and H, "Upload overlap complex, C(m1), C(m2), H"
            CO = load_complex(json.load(c_overlap))
            C1 = load_map_blocks(json.load(cm1))
            C2 = load_map_blocks(json.load(cm2))
            Hb = load_map_blocks(json.load(H))
            ok, res = overlap_test(CO, C1, C2, Hb)
            st.write("PASS" if ok else "FAIL")
            st.json(res)
    else:
        c_overlap = st.file_uploader("Overlap complex (JSON)", type=["json"], key="tri_c")
        Jfile = st.file_uploader("Triangle template J (JSON)", type=["json"], key="tri_J")
        if st.button("Run triangle coherence test"):
            assert c_overlap and Jfile, "Upload overlap complex and J template"
            CO = load_complex(json.load(c_overlap))
            J = json.load(Jfile)  # expects {k: {A:..., B:..., J:...}}
            ok, res = triangle_test(CO, J)
            st.write("PASS" if ok else "FAIL")
            st.json(res)

with tab3:
    st.header("Towers & Novelty (GF2)")
    st.markdown("Upload a list of move blocks to form the schedule. The app composes them and tracks certificate hashes.")
    reps = st.file_uploader("Representatives & degrees (JSON)", type=["json"], key="tw_reps")
    num = st.number_input("Number of steps in schedule", min_value=1, max_value=200, value=5, step=1)
    move_files = st.file_uploader("Upload move blocks for each step (JSON, in order)", type=["json"], accept_multiple_files=True, key="tw_moves")
    novelty_step = st.number_input("Novelty step (optional; 0 = none)", min_value=0, max_value=200, value=0, step=1)
    novelty_map = st.file_uploader("Novelty move blocks (JSON)", type=["json"], key="tw_nov")
    if st.button("Run tower"):
        assert reps and move_files and len(move_files) >= num, "Upload reps and at least 'Number of steps' move files"
        from otc.app_helpers import run_tower, load_map_blocks
        repsd = json.load(reps)
        seq = [load_map_blocks(json.load(f)) for f in move_files[:num]]
        base_hashes = run_tower(None, seq, repsd)
        df = pd.DataFrame(base_hashes)
        st.subheader("Baseline tower hashes")
        st.dataframe(df)
        if novelty_step > 0:
            assert novelty_map, "Upload novelty map for the chosen novelty step"
            nov = load_map_blocks(json.load(novelty_map))
            seq_nov = [(nov if i+1 == novelty_step else C) for i, C in enumerate(seq)]
            dfn = pd.DataFrame(run_tower(None, seq_nov, repsd))
            st.subheader("Tower with novelty injection")
            st.dataframe(dfn)
            # Compare and report first divergence
            div = None
            for i in range(min(len(df), len(dfn))):
                if df.loc[i, "hash"] != dfn.loc[i, "hash"]:
                    div = i+1; break
            if div:
                st.error(f"Novelty detected at step {div}.")
            else:
                st.info("No divergence detected in hashes.")

with tab4:
    st.markdown("""
### Notes
- **Z-lift** checks signed boundary and pairing. Homology transport remains GF(2) in this build.
- **Towers**: composition + hash of mapped reps per step; novelty replaces one step and reports first divergence.
- Use JSON schemas consistent with your 4D data (degree-indexed block matrices).
""")
