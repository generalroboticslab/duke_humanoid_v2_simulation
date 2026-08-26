"""Rewrite a committed cuRobo URDF's mesh paths to wherever the meshes are on THIS machine.

``export_mjspec_to_urdf.py`` writes ``<robot>_curobo.urdf`` with ABSOLUTE mesh filenames on
purpose: cuRobo's RobotBuilder is handed ``asset_root_path="/"``, so the filename in the URDF is
used verbatim. That is fine on the box that generated the file and broken everywhere else, which
made the whole cuRobo path (the Table IV benchmark, ``curobo_reach_verify.py``) unusable in a
fresh clone -- meshes resolve to a directory that does not exist. The committed URDFs currently
carry three different machines' prefixes: this repo's, a ``mjlab`` checkout's, and (for
``fourier_gr3``) the path of the laptop the file was exported on.

Rewriting at LOAD time rather than fixing the files is deliberate. The alternative, re-exporting
all eight URDFs with repo-relative paths, cannot express the ``g1`` meshes at all: they live inside
the installed ``mjlab`` package, not in this repo, so no single ``asset_root_path`` can cover both
roots. Resolving per-mesh here handles the mixed case and leaves the artifacts byte-identical to
what the exporter produced, so re-exporting stays a clean diff.

Two rules, and they cannot collide -- repo meshes sit under ``/asset/`` and mjlab's under
``/asset_zoo/``:

  * ``.../asset/<rest>``     -> ``<repo root>/asset/<rest>``
  * ``.../asset_zoo/<rest>`` -> ``<mjlab package dir>/asset_zoo/<rest>``

A URDF whose meshes all already exist is returned untouched, so the machine that generated it pays
nothing and no temp file appears.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import tempfile

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MESH_RE = re.compile(r'filename="([^"]+)"')


def _mjlab_root() -> pathlib.Path | None:
    """Directory of the installed ``mjlab`` package, or None if it is not importable."""
    try:
        import mjlab
    except ImportError:
        return None
    return pathlib.Path(mjlab.__file__).resolve().parent


def _relocate(filename: str) -> str:
    """Map one mesh path onto this tree, falling back to the committed path.

    The in-tree copy WINS over an absolute path that happens to exist. Preferring the committed
    path whenever it resolved would let an export sitting next to its generating checkout keep
    reading that checkout's meshes -- a release copy would silently render from the private tree it
    was built from and only break once that tree was gone, which is the failure this module exists
    to remove. Resolving against this file's own repo root instead makes each tree self-contained.
    """
    path = pathlib.Path(filename)
    if not path.is_absolute():
        return filename
    parts = path.parts
    if "asset_zoo" in parts:
        root = _mjlab_root()
        if root is not None:
            candidate = root.joinpath(*parts[parts.index("asset_zoo"):])
            if candidate.exists():
                return str(candidate)
    if "asset" in parts:
        # rindex: an absolute path can contain "asset" in a user directory too, and the LAST
        # occurrence is the one that starts the in-repo layout.
        idx = len(parts) - 1 - parts[::-1].index("asset")
        candidate = _REPO_ROOT.joinpath(*parts[idx:])
        if candidate.exists():
            return str(candidate)
    return filename


def localize_urdf(urdf_path: str) -> str:
    """Path to a URDF whose mesh refs resolve here: the original when it already does.

    Args:
        urdf_path: Committed ``<robot>_curobo.urdf`` carrying the exporting machine's paths.

    Returns:
        ``urdf_path`` unchanged when every mesh already points into this tree, else a rewritten copy
        under the system temp directory, named by a hash of the source path and content so
        concurrent sweeps (``dyn_sweep.py`` runs several processes per GPU) share one file instead
        of racing.
    """
    src = pathlib.Path(urdf_path)
    text = src.read_text()
    names = set(_MESH_RE.findall(text))
    # Every mesh is relocated, not only the ones that fail to resolve -- see `_relocate`.
    localized = {n: _relocate(n) for n in names}
    if all(v == n for n, v in localized.items()):
        return urdf_path

    unresolved = [n for n, v in localized.items() if v == n and not pathlib.Path(n).exists()]
    if unresolved:
        raise FileNotFoundError(
            f"{src.name}: {len(unresolved)} mesh path(s) could not be located on this machine, "
            f"e.g. {unresolved[0]!r}. Meshes are expected under '<repo>/asset/' or inside the "
            f"installed mjlab package; check that the asset directory for this robot was shipped.")

    out_text = _MESH_RE.sub(lambda m: f'filename="{localized.get(m.group(1), m.group(1))}"', text)
    key = hashlib.sha1(f"{src}\0{out_text}".encode()).hexdigest()[:16]
    dst = pathlib.Path(tempfile.gettempdir()) / f"{src.stem}.{key}.urdf"
    if not dst.exists():
        tmp = dst.with_suffix(f".urdf.{os.getpid()}")
        tmp.write_text(out_text)
        tmp.replace(dst)  # atomic: concurrent writers cannot hand cuRobo a half-written file
    return str(dst)
