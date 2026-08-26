"""Hardware readout (PROVENANCE — not part of the model build; see README.md).

Read the factory (per-unit) RGB<->depth extrinsics off a connected RealSense D436.

  IN    a physical D436 on USB (values below came from serial 408122071763)
  OUT   stdout; the color->depth translation is transcribed into
        head_camera_creation.RGB_C2D_T_OPT
  DEPS  pyrealsense2 + the plugged-in camera
  PINS  RGB_C2D_T_OPT only. Re-run ONLY when swapping to a different physical
        camera unit — the extrinsic is per-unit factory calibration, not a
        model-wide constant.


The RealSense SDK stores a UNIQUE, factory-calibrated rigid transform between the
color imager and the depth frame on every physical unit (calibration table in
firmware). The depth frame's origin is the LEFT INFRARED imager. Convention is the
optical frame: +X right, +Y down, +Z forward (out of the lens), in METERS.

What we want for the sim: the pose of the RGB optical centre expressed IN the depth
frame (= the "cam_front_center" reference, if that frame is defined as the depth /
left-IR origin in the same optical convention). That is exactly the color->depth
extrinsic:  p_depth = R_c2d @ p_color + t_c2d  =>  RGB origin (p_color=0) sits at
t_c2d in depth coords, and the RGB axes in depth coords are the columns of R_c2d.

So:   site pos  = t_c2d         (metres, relative to cam_front_center)
      site quat = quat(R_c2d)   (MuJoCo w x y z)

Run (camera plugged into THIS machine):
    ~/miniconda3/envs/mjhand/bin/python read_d436_extrinsics.py
"""
from __future__ import annotations
import numpy as np
import pyrealsense2 as rs

np.set_printoptions(suppress=True, precision=8)


def extr_to_Rt(e):
    """rs2_extrinsics -> (R 3x3, t 3). rotation is COLUMN-MAJOR in the SDK."""
    R = np.array(e.rotation, dtype=float).reshape(3, 3).T   # col-major -> row-major
    t = np.array(e.translation, dtype=float)
    return R, t


def mat_to_quat(R):
    """3x3 rotation -> unit quaternion (w, x, y, z) (MuJoCo order)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rot_angle_deg(R):
    """Geodesic rotation angle of R from identity, in degrees (sanity metric)."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def dump(name, R, t):
    print(f"\n  [{name}]")
    print(f"    translation (m) = [{t[0]:+.7f}, {t[1]:+.7f}, {t[2]:+.7f}]   |t|={np.linalg.norm(t)*1000:.3f} mm")
    print(f"    rotation angle from identity = {rot_angle_deg(R):.4f} deg")
    q = mat_to_quat(R)
    print(f"    quat (w x y z)  = [{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}]")
    return q


def main():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise SystemExit("No RealSense device found. Is the D436 plugged in?")
    d = devs[0]
    print("Device:")
    for k in (rs.camera_info.name, rs.camera_info.product_line,
              rs.camera_info.serial_number, rs.camera_info.firmware_version):
        try:
            print(f"  {str(k).split('.')[-1]:18s} {d.get_info(k)}")
        except Exception:
            pass

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth)
    cfg.enable_stream(rs.stream.color)
    profile = pipe.start(cfg)
    try:
        depth_p = profile.get_stream(rs.stream.depth)
        color_p = profile.get_stream(rs.stream.color)

        print("\n" + "=" * 68)
        print("RGB <-> DEPTH factory extrinsics (this unit)")
        print("convention: +X right, +Y down, +Z forward (out lens), metres")
        print("depth-frame origin = LEFT IR imager")
        print("=" * 68)

        R_c2d, t_c2d = extr_to_Rt(color_p.get_extrinsics_to(depth_p))
        R_d2c, t_d2c = extr_to_Rt(depth_p.get_extrinsics_to(color_p))

        q_c2d = dump("color -> depth  (USE THIS: RGB origin/axes in depth frame)", R_c2d, t_c2d)
        dump("depth -> color  (reference / inverse)", R_d2c, t_d2c)

        # optional: confirm depth origin == left IR (infra1) if infrared exposed
        try:
            ir_p = profile.get_stream(rs.stream.infrared, 1)
            R_ir, t_ir = extr_to_Rt(depth_p.get_extrinsics_to(ir_p))
            print(f"\n  [sanity] depth -> infrared(left): |t|={np.linalg.norm(t_ir)*1000:.4f} mm, "
                  f"angle={rot_angle_deg(R_ir):.4f} deg  (expect ~0 => depth origin IS left IR)")
        except Exception as e:
            print(f"\n  [sanity] infrared stream not queried ({e})")

        # bonus: color intrinsics
        ci = color_p.as_video_stream_profile().get_intrinsics()
        print(f"\n  [color intrinsics] {ci.width}x{ci.height}  fx={ci.fx:.3f} fy={ci.fy:.3f} "
              f"cx={ci.ppx:.3f} cy={ci.ppy:.3f}  model={ci.model}  coeffs={[round(c,5) for c in ci.coeffs]}")

        print("\n" + "=" * 68)
        print("READY TO PASTE — RGB optical site relative to cam_front_center")
        print("(only valid if cam_front_center == depth/left-IR origin, optical convention)")
        print("=" * 68)
        print(f'  <site name="..._rgb" '
              f'pos="{t_c2d[0]:.6f} {t_c2d[1]:.6f} {t_c2d[2]:.6f}" '
              f'quat="{q_c2d[0]:.6f} {q_c2d[1]:.6f} {q_c2d[2]:.6f} {q_c2d[3]:.6f}"/>')
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
