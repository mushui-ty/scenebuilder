"""Split merged visible PLY/GLB by per-view visibility sidecars (viewvis NPZ)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

VisibilityMode = Literal["view", "object"]

_PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
_PLY_VERTEX_SIZE = _PLY_VERTEX_DTYPE.itemsize


def read_scenebuilder_ply(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (missing end_header): {path}")
            header_lines.append(line.decode("ascii", errors="replace").rstrip("\n"))
            if line.strip() == b"end_header":
                break
        header_text = "\n".join(header_lines) + "\n"
        vertex_count = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
        raw = f.read(vertex_count * _PLY_VERTEX_SIZE)
    if len(raw) != vertex_count * _PLY_VERTEX_SIZE:
        raise ValueError(
            f"PLY vertex count mismatch: expected {vertex_count}, got {len(raw) // _PLY_VERTEX_SIZE} in {path}"
        )
    packed = np.frombuffer(raw, dtype=_PLY_VERTEX_DTYPE, count=vertex_count)
    points = np.column_stack([packed["x"], packed["y"], packed["z"]]).astype(np.float32)
    normals = np.column_stack([packed["nx"], packed["ny"], packed["nz"]]).astype(np.float32)
    colors = np.column_stack([packed["red"], packed["green"], packed["blue"]]).astype(np.uint8)
    return points, colors, normals


def write_scenebuilder_ply(
    path: str,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray | None = None,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    cols = np.clip(np.asarray(colors, dtype=np.uint8), 0, 255)
    if normals is None:
        nrm = np.zeros((len(pts), 3), dtype=np.float32)
    else:
        nrm = np.asarray(normals, dtype=np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    packed = np.empty(len(pts), dtype=_PLY_VERTEX_DTYPE)
    packed["x"] = pts[:, 0]
    packed["y"] = pts[:, 1]
    packed["z"] = pts[:, 2]
    packed["nx"] = nrm[:, 0]
    packed["ny"] = nrm[:, 1]
    packed["nz"] = nrm[:, 2]
    packed["red"] = cols[:, 0]
    packed["green"] = cols[:, 1]
    packed["blue"] = cols[:, 2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(packed.tobytes())


def _geometry_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return "ply"
    if ext == ".glb":
        return "glb"
    raise ValueError(f"Unsupported geometry file: {path} (expected .ply or .glb)")


def _output_stem(geometry_path: str) -> str:
    return os.path.splitext(os.path.basename(geometry_path))[0]


def _meta_json_path(geometry_dir: str, kind: str) -> str:
    if kind == "ply":
        return os.path.join(geometry_dir, "scene_visible_viewvis.json")
    return os.path.join(geometry_dir, "scene_visible_viewvis_tri.json")


def _default_npz_path(geometry_dir: str, kind: str, mode: VisibilityMode) -> str:
    if kind == "ply":
        name = "scene_visible_viewvis_point.npz" if mode == "view" else "scene_visible_viewvis_object.npz"
    else:
        name = (
            "scene_visible_viewvis_point_tri.npz"
            if mode == "view"
            else "scene_visible_viewvis_object_tri.npz"
        )
    return os.path.join(geometry_dir, name)


def resolve_viewvis_sidecar(geometry_path: str, mode: VisibilityMode) -> Tuple[str, List[str], Dict[str, Any]]:
    geometry_path = os.path.abspath(geometry_path)
    if not os.path.isfile(geometry_path):
        raise FileNotFoundError(geometry_path)

    kind = _geometry_kind(geometry_path)
    geometry_dir = os.path.dirname(geometry_path)
    view_dir = os.path.dirname(geometry_dir) if os.path.basename(geometry_dir) in ("pointcloud", "glb") else geometry_dir
    meta: Dict[str, Any] = {}
    meta_path = _meta_json_path(geometry_dir, kind)
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    key = "viewvis_point" if mode == "view" else "viewvis_object"
    npz_rel = meta.get(key)
    if npz_rel:
        npz_path = os.path.join(view_dir, npz_rel)
        if not os.path.isfile(npz_path):
            npz_path = os.path.join(geometry_dir, os.path.basename(npz_rel))
    else:
        npz_path = _default_npz_path(geometry_dir, kind, mode)

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Viewvis sidecar not found for mode={mode!r}: {npz_path}")

    data = np.load(npz_path)
    visibility = np.asarray(data["visibility"])
    if "camera_frames" in data:
        frames = [str(x) for x in np.asarray(data["camera_frames"]).tolist()]
    elif meta.get("camera_frames"):
        frames = [str(x) for x in meta["camera_frames"]]
    else:
        frames = [f"view_{i}" for i in range(int(visibility.shape[1]))]

    return npz_path, frames, meta


def load_visibility_matrix(geometry_path: str, mode: VisibilityMode) -> Tuple[np.ndarray, List[str], str]:
    npz_path, frames, _meta = resolve_viewvis_sidecar(geometry_path, mode)
    visibility = np.asarray(np.load(npz_path)["visibility"])
    if visibility.ndim != 2:
        raise ValueError(f"Expected 2D visibility matrix in {npz_path}, got shape {visibility.shape}")
    if visibility.shape[1] != len(frames):
        frames = [f"view_{i}" for i in range(int(visibility.shape[1]))]
    return visibility, frames, npz_path


def default_output_dir(geometry_path: str, mode: VisibilityMode) -> str:
    geometry_path = os.path.abspath(geometry_path)
    geometry_dir = os.path.dirname(geometry_path)
    stem = _output_stem(geometry_path)
    return os.path.join(geometry_dir, f"{stem}_by_{mode}")


def split_ply_by_views(
    geometry_path: str,
    mode: VisibilityMode,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    visibility, frames, npz_path = load_visibility_matrix(geometry_path, mode)
    points, colors, normals = read_scenebuilder_ply(geometry_path)
    if len(points) != visibility.shape[0]:
        raise ValueError(
            f"PLY points ({len(points)}) != visibility rows ({visibility.shape[0]}) for {geometry_path}"
        )

    out_dir = output_dir or default_output_dir(geometry_path, mode)
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(geometry_path)[1]
    per_view: List[Dict[str, Any]] = []

    for view_idx, frame in enumerate(frames):
        mask = visibility[:, view_idx].astype(bool)
        out_path = os.path.join(out_dir, f"{frame}{ext}")
        kept = int(np.count_nonzero(mask))
        if kept > 0:
            write_scenebuilder_ply(out_path, points[mask], colors[mask], normals[mask])
        else:
            write_scenebuilder_ply(
                out_path,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=np.float32),
            )
        per_view.append({"frame": frame, "path": os.path.basename(out_path), "kept": kept, "total": len(points)})

    manifest = {
        "input": os.path.abspath(geometry_path),
        "mode": mode,
        "visibility_sidecar": npz_path,
        "element_type": "point",
        "views": per_view,
        "output_dir": os.path.abspath(out_dir),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def split_glb_by_views(
    geometry_path: str,
    mode: VisibilityMode,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("GLB split requires trimesh: pip install trimesh") from exc

    visibility, frames, npz_path = load_visibility_matrix(geometry_path, mode)
    scene = trimesh.load(geometry_path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError(f"Expected GLB scene, got {type(scene)}")

    global_idx = 0
    chunks: List[Tuple[str, Any, slice]] = []
    for name, geom in scene.geometry.items():
        n_faces = len(geom.faces)
        chunks.append((name, geom, slice(global_idx, global_idx + n_faces)))
        global_idx += n_faces

    if global_idx != visibility.shape[0]:
        raise ValueError(
            f"GLB triangles ({global_idx}) != visibility rows ({visibility.shape[0]}) for {geometry_path}"
        )

    out_dir = output_dir or default_output_dir(geometry_path, mode)
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(geometry_path)[1]
    per_view: List[Dict[str, Any]] = []

    for view_idx, frame in enumerate(frames):
        view_mask = visibility[:, view_idx].astype(bool)
        out_scene = trimesh.Scene()
        kept_faces = 0
        for name, geom, sl in chunks:
            local_mask = view_mask[sl]
            if not np.any(local_mask):
                continue
            face_idx = np.flatnonzero(local_mask)
            sub = geom.submesh([face_idx], append=True, only_watertight=False)
            out_scene.add_geometry(sub, node_name=name, geom_name=name)
            kept_faces += len(face_idx)

        out_path = os.path.join(out_dir, f"{frame}{ext}")
        if kept_faces > 0:
            out_scene.export(out_path)
        else:
            trimesh.Scene().export(out_path)

        per_view.append(
            {"frame": frame, "path": os.path.basename(out_path), "kept_faces": kept_faces, "total_faces": global_idx}
        )

    manifest = {
        "input": os.path.abspath(geometry_path),
        "mode": mode,
        "visibility_sidecar": npz_path,
        "element_type": "triangle",
        "views": per_view,
        "output_dir": os.path.abspath(out_dir),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def split_geometry_by_views(
    geometry_path: str,
    mode: VisibilityMode,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    kind = _geometry_kind(geometry_path)
    if kind == "ply":
        return split_ply_by_views(geometry_path, mode, output_dir=output_dir)
    return split_glb_by_views(geometry_path, mode, output_dir=output_dir)
