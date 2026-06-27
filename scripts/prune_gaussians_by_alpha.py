from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras[int(parts[0])] = Camera(parts[1], int(parts[2]), int(parts[3]), np.array(parts[4:], dtype=np.float64))
    return cameras


def read_images(path: Path) -> list[ImagePose]:
    images: list[ImagePose] = []
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        images.append(
            ImagePose(
                image_id=int(parts[0]),
                qvec=np.array(parts[1:5], dtype=np.float64),
                tvec=np.array(parts[5:8], dtype=np.float64),
                camera_id=int(parts[8]),
                name=parts[9],
            )
        )
        if index < len(lines) and not lines[index].startswith("#"):
            index += 1
    return images


def project(points: np.ndarray, pose: ImagePose, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera.model != "PINHOLE":
        raise ValueError(f"Unsupported camera model: {camera.model}")
    fx, fy, cx, cy = camera.params
    rot = qvec_to_rotmat(pose.qvec)
    cam_points = points @ rot.T + pose.tvec
    z = cam_points[:, 2]
    valid = z > 1e-6
    u = fx * (cam_points[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (cam_points[:, 1] / np.maximum(z, 1e-6)) + cy
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return u, v, valid


def read_binary_ply(path: Path) -> tuple[list[str], list[tuple[str, str]], np.ndarray]:
    with path.open("rb") as handle:
        raw_header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing end_header in {path}")
            text = line.decode("ascii").rstrip("\n")
            raw_header.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in raw_header:
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")

        vertex_count = None
        vertex_props: list[tuple[str, str]] = []
        in_vertex = False
        for line in raw_header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and len(parts) == 3 and parts[0] == "property":
                ply_type = parts[1]
                if ply_type not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {ply_type}")
                vertex_props.append((parts[2], ply_type))

        if vertex_count is None:
            raise ValueError(f"Missing vertex element in {path}")

        dtype = np.dtype([(name, "<" + PLY_TYPES[ply_type] if PLY_TYPES[ply_type][-1] != "1" else PLY_TYPES[ply_type]) for name, ply_type in vertex_props])
        vertices = np.frombuffer(handle.read(dtype.itemsize * vertex_count), dtype=dtype, count=vertex_count).copy()
    return raw_header, vertex_props, vertices


def write_binary_ply(path: Path, raw_header: list[str], vertices: np.ndarray) -> None:
    header = []
    for line in raw_header:
        if line.startswith("element vertex "):
            header.append(f"element vertex {len(vertices)}")
        else:
            header.append(line)
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(vertices.tobytes())


def gaussian_xyz(vertices: np.ndarray) -> np.ndarray:
    names = vertices.dtype.names or ()
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"PLY is missing property {name}")
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)


def copy_model_if_needed(model: Path, output_model: Path | None) -> Path:
    if output_model is None:
        return model
    if output_model.exists():
        raise FileExistsError(output_model)
    shutil.copytree(model, output_model, copy_function=shutil.copy2)
    return output_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune trained Graphdeco Gaussians whose centers do not project into "
            "the foreground alpha masks often enough. This targets dark/black "
            "floating background splats around masked human subjects."
        )
    )
    parser.add_argument("--model", required=True, type=Path, help="Graphdeco output model directory.")
    parser.add_argument("--source", required=True, type=Path, help="COLMAP source dataset with RGBA images.")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output-model", type=Path, help="Copy the model here and replace only the selected PLY.")
    parser.add_argument("--alpha-threshold", type=int, default=32)
    parser.add_argument("--min-visible-views", type=int, default=1)
    parser.add_argument("--min-foreground-views", type=int, default=1)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    sparse = source / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    images = read_images(sparse / "images.txt")

    input_ply = args.model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    raw_header, _, vertices = read_binary_ply(input_ply)
    xyz = gaussian_xyz(vertices)

    visible_counts = np.zeros(len(xyz), dtype=np.int32)
    foreground_counts = np.zeros(len(xyz), dtype=np.int32)

    for pose in images:
        camera = cameras[pose.camera_id]
        alpha_path = source / "images" / pose.name
        alpha = np.asarray(Image.open(alpha_path).convert("RGBA"))[:, :, 3]
        u, v, valid = project(xyz, pose, camera)
        if not np.any(valid):
            continue
        valid_indices = np.flatnonzero(valid)
        ui = np.clip(np.rint(u[valid]).astype(np.int32), 0, camera.width - 1)
        vi = np.clip(np.rint(v[valid]).astype(np.int32), 0, camera.height - 1)
        hits = alpha[vi, ui] >= args.alpha_threshold
        visible_counts[valid_indices] += 1
        foreground_counts[valid_indices[hits]] += 1

    ratios = foreground_counts / np.maximum(visible_counts, 1)
    keep = (
        (visible_counts >= args.min_visible_views)
        & (foreground_counts >= args.min_foreground_views)
        & (ratios >= args.min_foreground_ratio)
    )

    stats = {
        "input_gaussians": int(len(vertices)),
        "kept_gaussians": int(keep.sum()),
        "removed_gaussians": int((~keep).sum()),
        "alpha_threshold": args.alpha_threshold,
        "min_visible_views": args.min_visible_views,
        "min_foreground_views": args.min_foreground_views,
        "min_foreground_ratio": args.min_foreground_ratio,
    }
    print(stats)
    if args.dry_run:
        return

    output_model = copy_model_if_needed(args.model, args.output_model)
    output_ply = output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    write_binary_ply(output_ply, raw_header, vertices[keep])


if __name__ == "__main__":
    main()
