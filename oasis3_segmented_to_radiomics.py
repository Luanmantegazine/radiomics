#!/usr/bin/env python3
"""
OASIS-3 smoke test:
FreeSurfer segmentation (aseg.mgz) -> hippocampus mask -> PyRadiomics.

What it does
------------
1. Reads a small CSV with OASIS-3 FreeSurfer IDs.
2. Uses the official NrgXnat/oasis-scripts downloader to fetch only those
   FreeSurfer processed sessions from NITRC-IR.
3. Finds T1.mgz and aseg.mgz in the downloaded FreeSurfer outputs.
4. Builds left, right and bilateral hippocampus masks from aseg labels:
      17 = Left-Hippocampus
      53 = Right-Hippocampus
5. Converts image/masks to NIfTI with nibabel.
6. Extracts a small set of Original-image radiomic features with PyRadiomics.
7. Saves one CSV with the extracted features.

This is a proof-of-concept/smoke test, NOT a final radiomics protocol.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


FREESURFER_REPO = "https://github.com/NrgXnat/oasis-scripts.git"
SESSION_RE = re.compile(r"OAS\d+_MR_d\d+")

ROI_LABELS = {
    "left_hippocampus": [17],
    "right_hippocampus": [53],
    "bilateral_hippocampus": [17, 53],
}


def require_dependencies():
    missing = []
    try:
        import nibabel  # noqa: F401
    except ImportError:
        missing.append("nibabel")
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import radiomics  # noqa: F401
    except ImportError:
        missing.append("pyradiomics")

    if missing:
        print("\nMissing Python dependencies:", ", ".join(missing), file=sys.stderr)
        print(
            "Install them with:\n"
            "  python -m pip install numpy pandas nibabel pyradiomics\n",
            file=sys.stderr,
        )
        sys.exit(2)


def read_ids(csv_path: Path, max_cases: int) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Create it with a header 'freesurfer_id' and 1-3 IDs, for example:\n"
            "freesurfer_id\n"
            "OAS30001_Freesurfer53_d0129\n"
            "OAS30001_Freesurfer53_d0757\n"
        )

    ids: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("The ID CSV is empty.")

    # Accept either a proper header or a simple one-ID-per-line file.
    start = 1 if rows[0] and rows[0][0].strip().lower() == "freesurfer_id" else 0

    for row in rows[start:]:
        if not row:
            continue
        fs_id = row[0].strip()
        if fs_id:
            ids.append(fs_id)
        if len(ids) >= max_cases:
            break

    if not ids:
        raise ValueError("No FreeSurfer IDs were found in the CSV.")

    return ids


def write_selected_csv(ids: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["freesurfer_id"])
        for fs_id in ids:
            writer.writerow([fs_id])


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found: {name}")


def ensure_oasis_scripts(repo_dir: Path) -> Path:
    downloader = repo_dir / "download_freesurfer" / "download_oasis_freesurfer.sh"
    if downloader.exists():
        return downloader

    ensure_command("git")
    print(f"\nCloning official OASIS downloader into {repo_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", FREESURFER_REPO, str(repo_dir)],
        check=True,
    )

    if not downloader.exists():
        raise RuntimeError("Official FreeSurfer downloader was not found after clone.")
    return downloader


def download_freesurfer(
    selected_csv: Path,
    download_dir: Path,
    nitrc_user: str,
    repo_dir: Path,
) -> None:
    ensure_command("bash")
    ensure_command("curl")
    ensure_command("unzip")

    downloader = ensure_oasis_scripts(repo_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting OASIS-3 FreeSurfer download.")
    print("The official downloader will ask for your NITRC-IR password.")
    print("The password is handled by the official shell script and is not saved here.\n")

    subprocess.run(
        [
            "bash",
            str(downloader),
            str(selected_csv.resolve()),
            str(download_dir.resolve()),
            nitrc_user,
        ],
        check=True,
    )


def infer_session_id(path: Path) -> str:
    for part in reversed(path.parts):
        if SESSION_RE.fullmatch(part):
            return part
    return path.parent.parent.name


def find_cases(download_dir: Path) -> list[tuple[str, Path, Path]]:
    cases = []
    for aseg_path in sorted(download_dir.rglob("aseg.mgz")):
        # Standard FreeSurfer structure: <subject>/mri/aseg.mgz
        mri_dir = aseg_path.parent
        t1_path = mri_dir / "T1.mgz"
        if not t1_path.exists():
            continue

        session_id = infer_session_id(aseg_path)
        cases.append((session_id, t1_path, aseg_path))

    return cases


def save_nifti_and_masks(
    session_id: str,
    t1_path: Path,
    aseg_path: Path,
    prepared_dir: Path,
) -> list[tuple[str, Path, Path, int]]:
    import nibabel as nib

    session_out = prepared_dir / session_id
    session_out.mkdir(parents=True, exist_ok=True)

    t1 = nib.load(str(t1_path))
    aseg = nib.load(str(aseg_path))

    if t1.shape[:3] != aseg.shape[:3]:
        raise ValueError(
            f"Geometry mismatch for {session_id}: "
            f"T1 shape={t1.shape}, aseg shape={aseg.shape}"
        )

    t1_data = np.asarray(t1.dataobj, dtype=np.float32)
    aseg_data = np.rint(np.asarray(aseg.dataobj)).astype(np.int32)

    image_nii = session_out / "T1.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, t1.affine), str(image_nii))

    outputs = []
    for roi_name, labels in ROI_LABELS.items():
        mask = np.isin(aseg_data, labels).astype(np.uint8)
        nvox = int(mask.sum())
        if nvox == 0:
            print(f"WARNING: {session_id} has no voxels for {roi_name}.")
            continue

        mask_nii = session_out / f"{roi_name}_mask.nii.gz"

        # Use the image affine so PyRadiomics sees image and mask in identical geometry.
        nib.save(nib.Nifti1Image(mask, t1.affine), str(mask_nii))
        outputs.append((roi_name, image_nii, mask_nii, nvox))

    return outputs


def make_extractor():
    from radiomics import featureextractor

    # Smoke-test parameters only.
    settings = {
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "correctMask": True,
        "label": 1,
        "enableCExtensions": False,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    for feature_class in (
        "firstorder",
        "shape",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
    ):
        extractor.enableFeatureClassByName(feature_class)

    # Default image type remains "Original"; no wavelet/LoG for this first test.
    return extractor


def extract_features(
    session_id: str,
    roi_name: str,
    image_path: Path,
    mask_path: Path,
    nvox: int,
    extractor,
) -> dict:
    result = extractor.execute(str(image_path), str(mask_path))

    row = {
        "session_id": session_id,
        "roi": roi_name,
        "mask_voxels": nvox,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
    }

    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        row[key] = value

    return row


def create_example_csv(path: Path) -> None:
    # These IDs are from the official NrgXnat/oasis-scripts example file.
    examples = [
        "OAS30001_Freesurfer53_d0129",
        "OAS30001_Freesurfer53_d0757",
    ]
    write_selected_csv(examples, path)
    print(f"Example CSV created: {path}")
    print("Review the IDs in NITRC-IR before downloading if needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Small OASIS-3 FreeSurfer -> hippocampus -> PyRadiomics smoke test."
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("freesurfer_ids.csv"),
        help="CSV containing a freesurfer_id column.",
    )
    parser.add_argument(
        "--nitrc-user",
        help="NITRC-IR username. Required unless --skip-download is used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("oasis3_radiomics_smoketest"),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=2,
        help="Maximum number of FreeSurfer sessions to download/process (default: 2).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and process existing FreeSurfer files under <out>/freesurfer.",
    )
    parser.add_argument(
        "--make-example-csv",
        action="store_true",
        help="Create a 2-session example freesurfer_ids.csv and exit.",
    )
    args = parser.parse_args()

    if args.make_example_csv:
        create_example_csv(args.ids)
        return

    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")

    require_dependencies()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = out_dir / "selected_freesurfer_ids.csv"
    download_dir = out_dir / "freesurfer"
    prepared_dir = out_dir / "prepared_nifti"
    repo_dir = out_dir / "_oasis_scripts"

    ids = read_ids(args.ids, args.max_cases)
    write_selected_csv(ids, selected_csv)

    print("\nSelected FreeSurfer IDs:")
    for fs_id in ids:
        print(f"  - {fs_id}")

    if not args.skip_download:
        if not args.nitrc_user:
            raise ValueError("--nitrc-user is required unless --skip-download is used.")
        download_freesurfer(
            selected_csv=selected_csv,
            download_dir=download_dir,
            nitrc_user=args.nitrc_user,
            repo_dir=repo_dir,
        )

    cases = find_cases(download_dir)
    if not cases:
        raise RuntimeError(
            f"No paired T1.mgz + aseg.mgz files were found under {download_dir}.\n"
            "Check whether the selected OASIS sessions have FreeSurfer processed outputs."
        )

    print(f"\nFound {len(cases)} usable FreeSurfer session(s).")
    extractor = make_extractor()

    rows = []
    for session_id, t1_path, aseg_path in cases[: args.max_cases]:
        print(f"\nProcessing {session_id}")
        print(f"  image: {t1_path}")
        print(f"  aseg : {aseg_path}")

        rois = save_nifti_and_masks(
            session_id=session_id,
            t1_path=t1_path,
            aseg_path=aseg_path,
            prepared_dir=prepared_dir,
        )

        for roi_name, image_nii, mask_nii, nvox in rois:
            print(f"  PyRadiomics: {roi_name} ({nvox} voxels)")
            row = extract_features(
                session_id=session_id,
                roi_name=roi_name,
                image_path=image_nii,
                mask_path=mask_nii,
                nvox=nvox,
                extractor=extractor,
            )
            rows.append(row)

    if not rows:
        raise RuntimeError("No radiomic features were extracted.")

    import pandas as pd

    df = pd.DataFrame(rows)
    output_csv = out_dir / "radiomics_features.csv"
    df.to_csv(output_csv, index=False)

    print("\nDONE")
    print(f"Radiomic feature table: {output_csv}")
    print(f"Prepared NIfTI files:   {prepared_dir}")
    print(f"Downloaded FreeSurfer:  {download_dir}")
    print(
        "\nReminder: bin width, normalization, resampling, ROI definitions and "
        "feature robustness must be defined rigorously before the real study."
    )


if __name__ == "__main__":
    main()
