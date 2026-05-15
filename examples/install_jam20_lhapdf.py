#!/usr/bin/env python3
"""Install JAM20-SIDIS PDF and FF sets for LHAPDF (required for SIDIS_masked_cc theory)."""
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SETS = ["JAM20-SIDIS_PDF_proton_nlo", "JAM20-SIDIS_FF_pion_nlo"]
REPO_ZIP = "https://github.com/QCDHUB/JAM20SIDIS/archive/refs/heads/main.zip"


def get_lhapdf_dir():
    if os.environ.get("LHAPDF_DATA_PATH"):
        return Path(os.environ["LHAPDF_DATA_PATH"])
    if os.environ.get("CONDA_PREFIX"):
        base = Path(os.environ["CONDA_PREFIX"]) / "share"
        for p in ["LHAPDF", "lhapdf"]:
            d = base / p
            if d.exists():
                return d
        return base / "LHAPDF"
    return Path("/usr/share/LHAPDF")


def main():
    lhapdf_dir = get_lhapdf_dir()
    lhapdf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Installing JAM20-SIDIS sets to: {lhapdf_dir}\n")

    # Download repo zip
    print("Downloading JAM20SIDIS from GitHub...")
    zip_path, _ = urllib.request.urlretrieve(REPO_ZIP)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract to temp, then copy each set
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                src_base = Path(tmp) / "JAM20SIDIS-main"
                for name in SETS:
                    dest = lhapdf_dir / name
                    if dest.exists():
                        print(f"  {name} already exists, skipping")
                        continue
                    src = src_base / name
                    if not src.exists():
                        print(f"  WARNING: {name} not found in archive")
                        continue
                    shutil.copytree(src, dest)
                    print(f"  Installed {name}")
    finally:
        os.unlink(zip_path)

    print("\nDone. For current session:")
    print(f"  export LHAPDF_DATA_PATH={lhapdf_dir}")
    print("\nThen re-run your training script.")


if __name__ == "__main__":
    main()
