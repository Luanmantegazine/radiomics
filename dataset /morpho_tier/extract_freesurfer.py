"""Fill the fs_* columns of the tier table from FreeSurfer outputs.

Mode A (raw stats):   --stats-root DIR
    Each FreeSurfer run is a folder under DIR containing stats/aseg.stats,
    stats/lh.aparc.stats and stats/rh.aparc.stats (Desikan-Killiany).
    Folder name must contain the subject id (OAS3xxxx) and the scan day (_dNNNN),
    e.g. OAS30001_Freesurfer53_d0129 -> matches session OAS30001_MR_d0129.
Mode B (exported table): --measures-csv FILE --map map.json
    map.json maps each fs_* column to the header name in FILE; FILE needs a
    column with the session id (--key-col, default session_id) or, if absent,
    subject + day are parsed from it the same way as in mode A.

Optionally --merge-into CONSOLIDATED.csv writes a copy with the fs_* columns
left-joined by session_id.
"""
import argparse, glob, json, os, re, sys
import pandas as pd
from common import ALL_FS_COLUMNS

KEY_RE = re.compile(r"(OAS3\d{4}).*?_d(\d{3,5})")

def key_of(text):
    m = KEY_RE.search(text)
    return (m.group(1), int(m.group(2))) if m else None

def parse_aseg(path):
    meas, vols, cols = {}, {}, None
    for line in open(path):
        if line.startswith("# Measure"):
            p = [x.strip() for x in line[len("# Measure"):].split(",")]
            meas[p[0]] = meas[p[1]] = float(p[3])  # long and short measure names
        elif line.startswith("# ColHeaders"):
            cols = line.split()[2:]
        elif not line.startswith("#") and cols:
            f = line.split()
            vols[f[cols.index("StructName")]] = float(f[cols.index("Volume_mm3")])
    return meas, vols

def parse_aparc(path):
    cols, rows = None, {}
    for line in open(path):
        if line.startswith("# ColHeaders"):
            cols = line.split()[2:]
        elif not line.startswith("#") and cols:
            f = line.split()
            rows[f[0]] = float(f[cols.index("ThickAvg")])
    return rows

def from_stats_dir(root):
    found = {}
    for aseg in glob.glob(os.path.join(root, "**", "stats", "aseg.stats"), recursive=True):
        sdir = os.path.dirname(aseg)
        run = os.path.relpath(sdir, root)
        k = key_of(run)
        if k is None:
            print(f"[warn] cannot parse subject/day from {run}", file=sys.stderr); continue
        lh, rh = (os.path.join(sdir, f"{s}.aparc.stats") for s in ("lh", "rh"))
        if not (os.path.exists(lh) and os.path.exists(rh)):
            print(f"[warn] missing aparc stats in {run}", file=sys.stderr); continue
        if k in found:
            print(f"[warn] duplicate FreeSurfer run for {k}: keeping first ({found[k]['_run']}), skipping {run}", file=sys.stderr); continue
        meas, vols = parse_aseg(aseg); L, R = parse_aparc(lh), parse_aparc(rh)
        found[k] = {
            "_run": run,
            "fs_left_hippocampus_mm3": vols.get("Left-Hippocampus"),
            "fs_right_hippocampus_mm3": vols.get("Right-Hippocampus"),
            "fs_left_amygdala_mm3": vols.get("Left-Amygdala"),
            "fs_right_amygdala_mm3": vols.get("Right-Amygdala"),
            "fs_left_inf_lat_vent_mm3": vols.get("Left-Inf-Lat-Vent"),
            "fs_right_inf_lat_vent_mm3": vols.get("Right-Inf-Lat-Vent"),
            "fs_left_lateral_ventricle_mm3": vols.get("Left-Lateral-Ventricle"),
            "fs_right_lateral_ventricle_mm3": vols.get("Right-Lateral-Ventricle"),
            "fs_lh_entorhinal_thickavg_mm": L.get("entorhinal"),
            "fs_rh_entorhinal_thickavg_mm": R.get("entorhinal"),
            "fs_lh_parahippocampal_thickavg_mm": L.get("parahippocampal"),
            "fs_rh_parahippocampal_thickavg_mm": R.get("parahippocampal"),
            "fs_etiv_mm3": meas.get("EstimatedTotalIntraCranialVol", meas.get("IntraCranialVol")),
            "fs_cortexvol_mm3": meas.get("CortexVol"),
        }
    return found

def from_measures_csv(path, mapping, key_col):
    t = pd.read_csv(path)
    found = {}
    for _, r in t.iterrows():
        k = key_of(str(r[key_col]))
        if k is None: continue
        found.setdefault(k, {c: r[src] for c, src in mapping.items() if src in t.columns})
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="morpho_tier_template.csv")
    ap.add_argument("--out", default="morpho_tier.csv")
    ap.add_argument("--stats-root"); ap.add_argument("--measures-csv")
    ap.add_argument("--map"); ap.add_argument("--key-col", default="session_id")
    ap.add_argument("--merge-into")
    a = ap.parse_args()
    if bool(a.stats_root) == bool(a.measures_csv):
        sys.exit("Give exactly one of --stats-root or --measures-csv")
    found = from_stats_dir(a.stats_root) if a.stats_root else \
        from_measures_csv(a.measures_csv, json.load(open(a.map)), a.key_col)
    tier = pd.read_csv(a.tier)
    filled = 0
    for i, r in tier.iterrows():
        rec = found.get(key_of(r.session_id))
        if rec is None: continue
        filled += 1
        for c in ALL_FS_COLUMNS:
            if rec.get(c) is not None: tier.at[i, c] = rec[c]
    tier.to_csv(a.out, index=False)
    print(f"{filled}/{len(tier)} sessions matched; missing per column:")
    print(tier[ALL_FS_COLUMNS].isna().sum().to_string())
    if a.merge_into:
        full = pd.read_csv(a.merge_into, low_memory=False)
        full = full.drop(columns=[c for c in ALL_FS_COLUMNS if c in full.columns])
        full = full.merge(tier[["session_id"] + ALL_FS_COLUMNS], on="session_id", how="left")
        dst = a.merge_into.replace(".csv", "_with_fs.csv"); full.to_csv(dst, index=False)
        print("wrote", dst)

if __name__ == "__main__":
    main()
