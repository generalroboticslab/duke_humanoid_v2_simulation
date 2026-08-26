"""
MuJoCo Viewer Test Script

Simple script to load and visualize the humanoid robot in MuJoCo.
Tests the validity of the generated MuJoCo XML file.

Usage:
    python test_mujoco_viewer.py
"""

import os
import mujoco
import mujoco.viewer


def main():
    # Get the XML file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "humanoid_v21.xml")

    print(f"Loading MuJoCo model from: {xml_path}")

    try:
        # Load the model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        print(f"✓ Model loaded successfully!")
        print(f"  - Number of bodies: {model.nbody}")
        print(f"  - Number of joints: {model.njnt}")
        print(f"  - Number of DOFs: {model.nv}")
        print(f"  - Number of geoms: {model.ngeom}")

        print("\nJoint names:")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = model.jnt_type[i]
            type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            print(f"  {i+1}. {joint_name} (type: {type_names.get(joint_type, 'unknown')})")

        print("\nLaunching interactive viewer...")
        print("Controls:")
        print("  - Left mouse: rotate")
        print("  - Right mouse: move")
        print("  - Scroll: zoom")
        print("  - Double-click: select body")
        print("  - Ctrl+Right click: apply force")
        print("  - Space: pause/resume")
        print("  - Backspace: reset")
        print("  - ESC or close window: exit")

        # Set initial position (floating base at reasonable height)
        # The free joint has 7 DOFs: 3 position (x,y,z) + 4 quaternion (w,x,y,z)
        data.qpos[2] = 0.8  # Set z-position to 0.8m above ground
        data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion (w,x,y,z)

        # Forward kinematics to update positions
        mujoco.mj_forward(model, data)

        # Launch the viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set initial camera
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.distance = 3.0

            # Simulation loop
            while viewer.is_running():
                # Keep the base_link fixed by resetting its position and velocity
                # Free joint DOFs: qpos[0:7] = [x, y, z, qw, qx, qy, qz]
                #                  qvel[0:6] = [vx, vy, vz, wx, wy, wz]
                data.qpos[0:3] = [0, 0, 0.0]  # Fixed position
                data.qpos[3:7] = [1, 0, 0, 0]  # Fixed orientation
                data.qvel[0:6] = 0  # Zero velocity

                # Step the simulation
                mujoco.mj_step(model, data)

                # Sync viewer
                viewer.sync()

    except Exception as e:
        print(f"\n✗ Error loading or viewing model:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nViewer closed.")
    return 0


if __name__ == "__main__":
    exit(main())
