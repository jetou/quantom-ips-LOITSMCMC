import os


def configure_lhapdf_paths(lhapdf):
    candidates = []

    env_path = os.environ.get("LHAPDF_DATA_PATH")
    if env_path:
        candidates.extend(path for path in env_path.split(":") if path)

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.extend(
            [
                os.path.join(conda_prefix, "share", "LHAPDF"),
                os.path.join(conda_prefix, "share", "lhapdf"),
            ]
        )

    candidates.extend(
        [
            "/usr/share/LHAPDF",
            "/usr/local/share/LHAPDF",
            "/w/jam-sciwork24/ccocuzza/lhapdf/python3/sets",
            "/home/jxu004/miniconda3/envs/tmd/share/LHAPDF",
            "/home/jxu004/miniconda3/envs/pyhigh2/share/LHAPDF",
        ]
    )

    seen = set()
    existing = []
    for path in candidates:
        if path and path not in seen and os.path.isdir(path):
            existing.append(path)
            seen.add(path)

    if existing:
        os.environ["LHAPDF_DATA_PATH"] = ":".join(existing)

    if hasattr(lhapdf, "pathsPrepend"):
        for path in reversed(existing):
            lhapdf.pathsPrepend(path)

    return existing
