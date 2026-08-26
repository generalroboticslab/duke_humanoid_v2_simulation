"""Split a merged OBJ into connected components and emit a colored MuJoCo scene.

Components above MIN_AREA are exported individually and assigned distinct colors
(golden-ratio hue stepping); the rest are merged into one gray small-parts mesh.
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import trimesh

MIN_AREA = 500.0  # mm^2 — components below this are grouped as small hardware


def distinct_color(i: int) -> tuple[float, float, float]:
    hue = (i * 0.618033988749895) % 1.0
    sat = 0.65 if i % 2 == 0 else 0.85
    val = 0.9 if i % 3 else 0.7
    return colorsys.hsv_to_rgb(hue, sat, val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", nargs="?",
                        default="meshes/yawpitch_cameras_twincities_high_res.obj")
    parser.add_argument("--out_dir", default="meshes/components_high_res")
    parser.add_argument("--xml", default="view_high_res_colored.xml")
    args = parser.parse_args()

    mesh = trimesh.load(args.obj, process=False)
    parts = sorted(mesh.split(only_watertight=False), key=lambda p: -p.area)
    main_parts = [p for p in parts if p.area >= MIN_AREA]
    small = [p for p in parts if p.area < MIN_AREA]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for i, part in enumerate(main_parts):
        name = f"part_{i:02d}"
        part.export(out_dir / f"{name}.obj")
        names.append(name)
    if small:
        merged = trimesh.util.concatenate(small)
        merged.export(out_dir / "small_parts.obj")

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2 * 0.001  # mm -> m, recenter xy on floor

    assets, geoms = [], []
    for i, name in enumerate(names):
        r, g, b = distinct_color(i)
        assets.append(
            f'    <mesh name="{name}" file="{out_dir.name}/{name}.obj"'
            f' scale="0.001 0.001 0.001"/>')
        geoms.append(
            f'      <geom type="mesh" mesh="{name}" material="part"'
            f' rgba="{r:.3f} {g:.3f} {b:.3f} 1" contype="0" conaffinity="0"/>')
    if small:
        assets.append(
            f'    <mesh name="small_parts" file="{out_dir.name}/small_parts.obj"'
            f' scale="0.001 0.001 0.001"/>')
        geoms.append(
            '      <geom type="mesh" mesh="small_parts" material="part"'
            ' rgba="0.45 0.45 0.45 1" contype="0" conaffinity="0"/>')

    xml = f"""<mujoco model="yawpitch_cameras_twincities_colored">
  <compiler meshdir="meshes"/>

  <asset>
{chr(10).join(assets)}
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.5 0.6" rgb2="0.1 0.1 0.15"
             width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.25 0.3"
             rgb2="0.3 0.35 0.4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="part" specular="0.5" shininess="0.4"/>
  </asset>

  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.25 0.25 0.25"/>
    <quality shadowsize="4096"/>
  </visual>

  <worldbody>
    <light pos="0.4 -0.4 0.8" dir="-0.4 0.4 -0.8" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="-0.4 0.4 0.6" dir="0.4 -0.4 -0.6" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid"/>
    <body name="yawpitch" pos="{-center[0]:.4f} {-center[1]:.4f} 0">
{chr(10).join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    Path(args.xml).write_text(xml)
    print(f"{len(names)} main components + {len(small)} small parts -> {args.xml}")


if __name__ == "__main__":
    main()
