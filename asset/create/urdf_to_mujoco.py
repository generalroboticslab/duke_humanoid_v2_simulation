"""
URDF to MuJoCo XML Converter

This module converts URDF robot descriptions to MuJoCo XML format.
It handles the conversion of links, joints, geometries, and inertial properties.

Key conversions:
    - URDF links → MuJoCo bodies
    - URDF joints → MuJoCo joints (with different syntax)
    - URDF collision geometries → MuJoCo geoms
    - URDF visual geometries → MuJoCo visual geoms
    - Joint limits → MuJoCo ranges
    - Effort limits → MuJoCo actuators
    - Sites for IMU and contact sensing
    - Sensors (gyro, velocimeter, accelerometer, etc.)

Usage:
    from urdf_to_mujoco import urdf_to_mujoco

    # Basic usage with defaults (humanoid robot)
    urdf_to_mujoco(robot_obj, "output.xml")

    # Custom configuration
    urdf_to_mujoco(
        robot_obj, "output.xml",
        imu_site_name="pelvis_imu",
        contact_body_names=["left_foot", "right_foot", "left_hand", "right_hand"],
        sensors=[
            {"type": "gyro", "name": "pelvis_gyro", "site": "pelvis_imu"},
            {"type": "velocimeter", "name": "pelvis_vel", "site": "pelvis_imu"},
            {"type": "touch", "name": "foot_contact", "site": "left_foot"},
        ]
    )

    # Disable auto-generated sites/sensors
    urdf_to_mujoco(robot_obj, "output.xml", imu_site_name=None,
                   contact_body_names=[], sensors=[])
"""

import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_array(text: str) -> List[float]:
    """Parse space-separated numeric string to list of floats."""
    return [float(x) for x in text.strip().split()]


def format_array(arr: List[float], precision: int = 6) -> str:
    """Format array as space-separated string."""
    return " ".join([f"{x:.{precision}g}" for x in arr])


def euler_to_quat(rpy: List[float]) -> List[float]:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    MuJoCo uses w x y z order.
    """
    roll, pitch, yaw = rpy

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def convert_geometry_to_mujoco(geom_elem: ET.Element, origin_elem: Optional[ET.Element],
                               is_collision: bool = True, material_name: Optional[str] = None,
                               cylinder_as_capsule: bool = False) -> Optional[ET.Element]:
    """
    Convert URDF geometry element to MuJoCo geom element.

    Args:
        geom_elem: URDF <geometry> element
        origin_elem: URDF <origin> element (optional)
        is_collision: Whether this is a collision geometry
        material_name: Material name for visual geometries

    Returns:
        MuJoCo <geom> element or None if conversion not supported
    """
    geom = ET.Element("geom")

    # Set class based on collision vs visual
    if is_collision:
        geom.set("class", "collision")
    else:
        geom.set("class", "visual")
        if material_name:
            geom.set("material", material_name)

    # Parse origin
    pos = [0, 0, 0]
    quat = [1, 0, 0, 0]  # w x y z
    if origin_elem is not None:
        if origin_elem.get("xyz"):
            pos = parse_array(origin_elem.get("xyz"))
        if origin_elem.get("rpy"):
            rpy = parse_array(origin_elem.get("rpy"))
            quat = euler_to_quat(rpy)

    # Only set pos if non-zero
    if not np.allclose(pos, [0, 0, 0], atol=1e-9):
        geom.set("pos", format_array(pos))
    # Only set quat if it's not identity
    if not np.allclose(quat, [1, 0, 0, 0], atol=1e-9):
        geom.set("quat", format_array(quat))

    # Convert geometry type
    for child in geom_elem:
        if child.tag == "box":
            geom.set("type", "box")
            size = parse_array(child.get("size"))
            # MuJoCo box uses half-sizes
            half_size = [s / 2.0 for s in size]
            geom.set("size", format_array(half_size))
            return geom

        elif child.tag == "cylinder":
            geom.set("type", "capsule" if cylinder_as_capsule else "cylinder")
            radius = float(child.get("radius"))
            length = float(child.get("length"))
            # MuJoCo cylinder/capsule both use half-length
            geom.set("size", f"{radius} {length/2.0}")
            return geom

        elif child.tag == "capsule":
            # odio_urdf Capsule element → MuJoCo capsule (hemispherical ends,
            # no length inflation vs cylinder).
            geom.set("type", "capsule")
            radius = float(child.get("radius"))
            length = float(child.get("length"))
            # MuJoCo capsule size = (radius, half-length)
            geom.set("size", f"{radius} {length/2.0}")
            return geom

        elif child.tag == "sphere":
            geom.set("type", "sphere")
            radius = float(child.get("radius"))
            geom.set("size", str(radius))
            return geom

        elif child.tag == "mesh":
            # For visual geoms with class, type is already set in default
            if is_collision:
                geom.set("type", "mesh")
            filename = child.get("filename")
            # Remove leading path and keep only filename
            if filename:
                geom.set("mesh", filename.replace("meshes/", "").replace(".stl", ""))
            # Note: scale is handled in the mesh asset definition, not on the geom
            return geom

    return None


def build_link_tree(urdf_root: ET.Element) -> Dict[str, List[Tuple[str, ET.Element]]]:
    """
    Build a tree structure showing parent-child relationships.

    Returns:
        Dictionary mapping parent link names to list of (child_name, joint_element) tuples
    """
    tree = {}

    for joint in urdf_root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")

        if parent_elem is not None and child_elem is not None:
            parent_name = parent_elem.get("link")
            child_name = child_elem.get("link")

            if parent_name not in tree:
                tree[parent_name] = []
            tree[parent_name].append((child_name, joint))

    return tree


def convert_link_to_mujoco_body(link_elem: ET.Element, joint_elem: Optional[ET.Element],
                                urdf_root: ET.Element, link_tree: Dict,
                                indent: int = 2,
                                cylinder_as_capsule: bool = False) -> ET.Element:
    """
    Recursively convert URDF link and its children to MuJoCo body elements.

    Args:
        link_elem: URDF <link> element
        joint_elem: URDF <joint> element connecting to parent (None for root)
        urdf_root: Root URDF element to look up child links
        link_tree: Dictionary of parent-child relationships
        indent: Current indentation level

    Returns:
        MuJoCo <body> element with nested children
    """
    body = ET.Element("body")
    link_name = link_elem.get("name")
    body.set("name", link_name)

    # Set position and orientation from joint
    if joint_elem is not None:
        origin = joint_elem.find("origin")
        if origin is not None:
            if origin.get("xyz"):
                pos = parse_array(origin.get("xyz"))
                body.set("pos", format_array(pos))
            if origin.get("rpy"):
                rpy = parse_array(origin.get("rpy"))
                quat = euler_to_quat(rpy)
                if not np.allclose(quat, [1, 0, 0, 0]):
                    body.set("quat", format_array(quat))

    # Add inertial properties
    inertial = link_elem.find("inertial")
    if inertial is not None:
        mass_elem = inertial.find("mass")
        if mass_elem is not None:
            mass = float(mass_elem.get("value"))
            inertial_body = ET.SubElement(body, "inertial")

            # Get inertial origin
            inertia_origin = inertial.find("origin")
            inertial_pos = [0, 0, 0]
            inertial_quat = [1, 0, 0, 0]
            if inertia_origin is not None:
                if inertia_origin.get("xyz"):
                    inertial_pos = parse_array(inertia_origin.get("xyz"))
                if inertia_origin.get("rpy"):
                    inertial_rpy = parse_array(inertia_origin.get("rpy"))
                    inertial_quat = euler_to_quat(inertial_rpy)

            # Set pos (required by MuJoCo)
            inertial_body.set("pos", format_array(inertial_pos))

            # Set quat if non-identity
            if not np.allclose(inertial_quat, [1, 0, 0, 0], atol=1e-9):
                inertial_body.set("quat", format_array(inertial_quat))

            inertial_body.set("mass", str(mass))

            # Get inertia
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.get("ixx", 0))
                iyy = float(inertia_elem.get("iyy", 0))
                izz = float(inertia_elem.get("izz", 0))
                ixy = float(inertia_elem.get("ixy", 0))
                ixz = float(inertia_elem.get("ixz", 0))
                iyz = float(inertia_elem.get("iyz", 0))

                # MuJoCo uses diagonal and off-diagonal separately
                inertial_body.set("diaginertia", format_array([ixx, iyy, izz]))
                if not np.allclose([ixy, ixz, iyz], [0, 0, 0], atol=1e-9):
                    # Note: MuJoCo may have different off-diagonal ordering
                    inertial_body.set("fullinertia", format_array([ixx, iyy, izz, ixy, ixz, iyz]))
                    # Remove diaginertia if using fullinertia
                    del inertial_body.attrib["diaginertia"]

    # Add collision geometries
    collision_idx = 0
    for collision in link_elem.findall("collision"):
        geom_elem = collision.find("geometry")
        origin_elem = collision.find("origin")
        
        # Check if the collision element has the special "capsule_collision" name.
        # This is a flag passed from urdf_helper.py when use_capsule_collision=True
        # to ensure that MJCF uses a capsule instead of a cylinder, while
        # preserving the cylinder tag in the standard URDF output for compatibility.
        force_capsule = (collision.get("name") == "capsule_collision")
        
        if geom_elem is not None:
            mj_geom = convert_geometry_to_mujoco(geom_elem, origin_elem, is_collision=True,
                                                   cylinder_as_capsule=cylinder_as_capsule or force_capsule)
            if mj_geom is not None:
                # Add index if there are multiple collisions
                if len(link_elem.findall("collision")) > 1:
                    mj_geom.set("name", f"{link_name}_collision{collision_idx}")
                    collision_idx += 1
                else:
                    mj_geom.set("name", f"{link_name}_collision")
                body.append(mj_geom)

    # Add visual geometries
    visual_idx = 0
    for visual in link_elem.findall("visual"):
        geom_elem = visual.find("geometry")
        origin_elem = visual.find("origin")
        material_name = None

        # Try to get material name
        material = visual.find("material")
        if material is not None:
            material_name = material.get("name")

        if geom_elem is not None:
            mj_geom = convert_geometry_to_mujoco(geom_elem, origin_elem, is_collision=False,
                                                   material_name=material_name,
                                                   cylinder_as_capsule=cylinder_as_capsule)
            if mj_geom is not None:
                # Add index if there are multiple visuals
                if len(link_elem.findall("visual")) > 1:
                    mj_geom.set("name", f"{link_name}_visual{visual_idx}")
                    visual_idx += 1
                else:
                    mj_geom.set("name", f"{link_name}_visual")
                body.append(mj_geom)

    # Add joint if not root
    if joint_elem is not None:
        joint_type = joint_elem.get("type")

        if joint_type in ["revolute", "continuous"]:
            mj_joint = ET.SubElement(body, "joint")
            joint_name = joint_elem.get("name")
            mj_joint.set("name", joint_name)
            mj_joint.set("type", "hinge")

            # Get axis
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis = parse_array(axis_elem.get("xyz"))
                mj_joint.set("axis", format_array(axis))

            # Get limits
            limit_elem = joint_elem.find("limit")
            if limit_elem is not None and joint_type == "revolute":
                lower = float(limit_elem.get("lower"))
                upper = float(limit_elem.get("upper"))
                mj_joint.set("range", f"{lower} {upper}")

                # Store effort and velocity for actuator creation later
                effort = float(limit_elem.get("effort", 0))
                velocity = float(limit_elem.get("velocity", 0))
                mj_joint.set("_effort", str(effort))  # Temporary attribute
                mj_joint.set("_velocity", str(velocity))  # Temporary attribute

        elif joint_type == "prismatic":
            mj_joint = ET.SubElement(body, "joint")
            joint_name = joint_elem.get("name")
            mj_joint.set("name", joint_name)
            mj_joint.set("type", "slide")

            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis = parse_array(axis_elem.get("xyz"))
                mj_joint.set("axis", format_array(axis))

            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.get("lower"))
                upper = float(limit_elem.get("upper"))
                mj_joint.set("range", f"{lower} {upper}")

        elif joint_type == "fixed":
            # Fixed joints don't need a joint element in MuJoCo
            pass

    # Recursively add children
    if link_name in link_tree:
        for child_name, child_joint in link_tree[link_name]:
            # Find child link element
            child_link = None
            for link in urdf_root.findall("link"):
                if link.get("name") == child_name:
                    child_link = link
                    break

            if child_link is not None:
                child_body = convert_link_to_mujoco_body(
                    child_link, child_joint, urdf_root, link_tree, indent + 2,
                    cylinder_as_capsule=cylinder_as_capsule
                )
                body.append(child_body)

    return body


def extract_actuators(body: ET.Element, actuators: List[Tuple[str, float, float]]):
    """
    Recursively extract joint information for actuator creation.

    Args:
        body: MuJoCo body element
        actuators: List to append (joint_name, effort, velocity) tuples
    """
    for joint in body.findall("joint"):
        if "_effort" in joint.attrib:
            joint_name = joint.get("name")
            effort = float(joint.get("_effort"))
            velocity = float(joint.get("_velocity"))
            actuators.append((joint_name, effort, velocity))

            # Remove temporary attributes
            del joint.attrib["_effort"]
            del joint.attrib["_velocity"]

    # Recurse to children
    for child_body in body.findall("body"):
        extract_actuators(child_body, actuators)


def extract_meshes(urdf_root: ET.Element) -> List[Tuple[str, str, Optional[List[float]]]]:
    """
    Extract all unique mesh filenames from URDF.

    Returns:
        List of tuples: (mesh_name, filename, scale)
    """
    meshes = {}

    for link in urdf_root.findall("link"):
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename:
                        # Extract just the filename without path or extension
                        mesh_name = filename.replace("meshes/", "").replace(".stl", "")
                        scale = None
                        if mesh.get("scale"):
                            scale = parse_array(mesh.get("scale"))
                        if mesh_name not in meshes:
                            meshes[mesh_name] = (mesh_name, filename, scale)

        for collision in link.findall("collision"):
            geom = collision.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename:
                        mesh_name = filename.replace("meshes/", "").replace(".stl", "")
                        scale = None
                        if mesh.get("scale"):
                            scale = parse_array(mesh.get("scale"))
                        if mesh_name not in meshes:
                            meshes[mesh_name] = (mesh_name, filename, scale)

    return sorted(list(meshes.values()))


def add_site_to_body(body: ET.Element, name: str, pos: str = "0 0 0", size: str = "0.01"):
    """
    Add a site element to a body.

    Args:
        body: The body element to add the site to
        name: Name of the site
        pos: Position string "x y z"
        size: Size of the site for visualization
    """
    site = ET.Element("site")
    site.set("name", name)
    site.set("pos", pos)
    site.set("size", size)

    # Insert before child bodies
    insert_idx = len(body)
    for i, child in enumerate(body):
        if child.tag == "body":
            insert_idx = i
            break
    body.insert(insert_idx, site)


def add_sites_and_sensors(
    root_body: ET.Element,
    worldbody: ET.Element,
    mujoco: ET.Element,
    imu_site_name: str = "imu_site",
    contact_body_names: Optional[List[str]] = None,
    sensors: Optional[List[Dict]] = None
):
    """
    Add sites and sensors to the MuJoCo model.

    This function adds:
    1. An IMU site at the origin of the root body (for gyro/velocimeter/accelerometer)
    2. Contact sites at the center of collision geometries for specified bodies
    3. A sensor section with configured sensors

    Args:
        root_body: The root body element (e.g., base_link)
        worldbody: The worldbody element containing all bodies
        mujoco: The root mujoco element to add sensor section to
        imu_site_name: Name for the IMU site on root body. Set to None or "" to skip.
        contact_body_names: List of body names to add contact sites.
            Sites are placed at the position of the body's first collision geom.
            This is useful for contact detection or as reference points.
        sensors: List of sensor configurations. Each dict should have:
            - "type": MuJoCo sensor type string
            - "name": unique sensor name
            - "site": (optional) site name for site-based sensors
            - "body": (optional) body name for body-based sensors

    Example:
        add_sites_and_sensors(
            root_body, worldbody, mujoco,
            imu_site_name="torso_imu",
            contact_body_names=["left_foot", "right_foot"],
            sensors=[
                {"type": "gyro", "name": "imu_gyro", "site": "torso_imu"},
                {"type": "touch", "name": "lfoot_touch", "site": "left_foot"},
            ]
        )
    """
    # Add IMU site to root body at origin
    if imu_site_name:
        add_site_to_body(root_body, imu_site_name, "0 0 0")

    # Add contact sites to specified bodies
    if contact_body_names:
        for body in worldbody.findall(".//body"):
            body_name = body.get("name", "")
            if body_name in contact_body_names:
                # Average all collision geom positions to get foot centroid
                collision_geoms = [g for g in body.findall("geom") if g.get("class") == "collision"]
                if collision_geoms:
                    positions = [parse_array(g.get("pos", "0 0 0")) for g in collision_geoms]
                    avg_pos = [sum(p[i] for p in positions) / len(positions) for i in range(3)]
                    add_site_to_body(body, body_name, format_array(avg_pos))

    # Add sensors
    if sensors:
        sensor_section = ET.SubElement(mujoco, "sensor")
        for sensor_cfg in sensors:
            sensor_elem = ET.SubElement(sensor_section, sensor_cfg["type"])
            sensor_elem.set("name", sensor_cfg["name"])
            if "site" in sensor_cfg:
                sensor_elem.set("site", sensor_cfg["site"])
            if "body" in sensor_cfg:
                sensor_elem.set("body", sensor_cfg["body"])


# =============================================================================
# Default Configurations
# =============================================================================

# Default sensor configuration for humanoid robots.
# Each sensor dict requires:
#   - type: MuJoCo sensor type (gyro, velocimeter, accelerometer, touch,
#           force, torque, framepos, framequat, subtreecom, subtreeangmom, etc.)
#   - name: unique sensor name
#   - site: (for site-based sensors) name of the site to attach to
#   - body: (for body-based sensors) name of the body to attach to
#
# Common sensor types:
#   - gyro: angular velocity in site frame (3D)
#   - velocimeter: linear velocity in site frame (3D)
#   - accelerometer: linear acceleration in site frame (3D)
#   - touch: contact normal force (scalar)
#   - force: 3D force at site
#   - torque: 3D torque at site
#   - framepos: position of frame origin (3D)
#   - framequat: orientation of frame (4D quaternion)
#   - subtreeangmom: angular momentum of subtree (3D)
#
DEFAULT_HUMANOID_SENSORS = [
    {"type": "gyro", "name": "imu_ang_vel", "site": "imu_site"},
    {"type": "velocimeter", "name": "imu_lin_vel", "site": "imu_site"},
    {"type": "accelerometer", "name": "imu_lin_acc", "site": "imu_site"},
    {"type": "subtreeangmom", "name": "root_angmom", "body": "base_link"},
]

# Default contact bodies for humanoid robots.
# Sites will be created at the center of each body's collision geometry.
# These sites can be used for contact sensors or as reference points.
DEFAULT_HUMANOID_CONTACT_BODIES = ["foot_L", "foot_R"]


def urdf_to_mujoco(
    robot_obj,
    output_path: str,
    include_actuators: bool = False,
    cylinder_as_capsule: bool = False,
    imu_site_name: str = "imu_site",
    contact_body_names: Optional[List[str]] = None,
    sensors: Optional[List[Dict]] = None
):
    """
    Convert odio_urdf Robot object to MuJoCo XML format.

    This is the main conversion function that:
    1. Parses the URDF robot structure
    2. Creates MuJoCo XML with bodies, joints, geoms, and inertials
    3. Adds default visual/collision classes
    4. Adds light and camera to root body
    5. Adds freejoint for floating base robots
    6. Adds IMU and contact sites
    7. Adds sensor section
    8. Optionally adds actuator section

    Args:
        robot_obj: odio_urdf Robot object to convert
        output_path: Path to write the MuJoCo XML file
        include_actuators: Whether to include actuator section with motors.
            Motors are created from joint effort limits in URDF.
        imu_site_name: Name for IMU site on root body.
            Set to None or "" to skip IMU site creation.
            Default: "imu_site"
        contact_body_names: List of body names to add contact sites.
            Sites are placed at collision geom positions.
            Default: DEFAULT_HUMANOID_CONTACT_BODIES (["foot_L", "foot_R"])
        sensors: List of sensor configurations.
            Each dict needs "type", "name", and "site" or "body".
            Default: DEFAULT_HUMANOID_SENSORS

    Note:
        - Quaternions use MuJoCo convention: (w, x, y, z)
        - Joint angles are in radians
        - Meshes are expected in a "meshes/" subdirectory

    Example:
        # Convert with all defaults (humanoid robot)
        urdf_to_mujoco(robot, "robot.xml")

        # Convert quadruped with custom contact bodies
        urdf_to_mujoco(
            robot, "robot.xml",
            contact_body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            sensors=[
                {"type": "gyro", "name": "imu_gyro", "site": "imu_site"},
                {"type": "velocimeter", "name": "imu_vel", "site": "imu_site"},
            ]
        )

        # Convert without any auto-generated sites/sensors
        urdf_to_mujoco(robot, "robot.xml",
                       imu_site_name=None, contact_body_names=[], sensors=[])
    """
    # Use defaults if not specified
    if contact_body_names is None:
        contact_body_names = DEFAULT_HUMANOID_CONTACT_BODIES
    if sensors is None:
        sensors = DEFAULT_HUMANOID_SENSORS
    # Convert robot to URDF string and parse
    urdf_str = str(robot_obj)
    urdf_root = ET.fromstring(urdf_str)

    # Create MuJoCo root
    mujoco = ET.Element("mujoco")
    mujoco.set("model", urdf_root.get("name", "robot"))

    # Add compiler settings
    compiler = ET.SubElement(mujoco, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "meshes")
    compiler.set("autolimits", "true")

    # Add default classes for visual and collision
    default = ET.SubElement(mujoco, "default")
    robot_class = ET.SubElement(default, "default")
    robot_class.set("class", urdf_root.get("name", "robot"))

    # Visual default
    visual_default = ET.SubElement(robot_class, "default")
    visual_default.set("class", "visual")
    visual_geom = ET.SubElement(visual_default, "geom")
    visual_geom.set("group", "2")
    visual_geom.set("type", "mesh")
    visual_geom.set("contype", "0")
    visual_geom.set("conaffinity", "0")
    visual_geom.set("material", "silver")

    # Collision default
    collision_default = ET.SubElement(robot_class, "default")
    collision_default.set("class", "collision")
    collision_geom = ET.SubElement(collision_default, "geom")
    collision_geom.set("group", "3")
    collision_geom.set("contype", "1")
    collision_geom.set("conaffinity", "1")

    # Add assets (materials and meshes)
    asset = ET.SubElement(mujoco, "asset")

    # Extract and add materials from URDF
    materials_added = set()
    for material in urdf_root.findall(".//material"):
        mat_name = material.get("name")
        if mat_name and mat_name not in materials_added:
            color_elem = material.find("color")
            if color_elem is not None:
                rgba = color_elem.get("rgba")
                if rgba:
                    mat_elem = ET.SubElement(asset, "material")
                    mat_elem.set("name", mat_name)
                    mat_elem.set("rgba", rgba)
                    materials_added.add(mat_name)

    # Add meshes
    meshes = extract_meshes(urdf_root)
    for mesh_name, filename, scale in meshes:
        mesh_elem = ET.SubElement(asset, "mesh")
        mesh_elem.set("name", mesh_name)
        # Remove meshes/ prefix since meshdir is already set
        mesh_elem.set("file", filename.replace("meshes/", ""))
        if scale is not None:
            # MuJoCo requires 3 values for scale (x, y, z)
            mesh_elem.set("scale", format_array(scale))

    # Build link tree
    link_tree = build_link_tree(urdf_root)

    # Find root link (base_link or first link with no parent)
    root_link = None
    all_children = set()
    for children in link_tree.values():
        for child_name, _ in children:
            all_children.add(child_name)

    for link in urdf_root.findall("link"):
        link_name = link.get("name")
        if link_name not in all_children or link_name == "base_link":
            root_link = link
            break

    if root_link is None and len(list(urdf_root.findall("link"))) > 0:
        root_link = urdf_root.findall("link")[0]

    # Create worldbody
    worldbody = ET.SubElement(mujoco, "worldbody")

    # Convert root link and its children
    if root_link is not None:
        root_link_name = root_link.get("name")

        # Add root body with childclass
        root_body = convert_link_to_mujoco_body(root_link, None, urdf_root, link_tree, indent=2,
                                                  cylinder_as_capsule=cylinder_as_capsule)
        root_body.set("childclass", urdf_root.get("name", "robot"))

        # Add light to root body
        light = ET.Element("light")
        light.set("pos", "0 0 2")
        light.set("mode", "trackcom")
        root_body.insert(0, light)

        # Add cameras to root body
        camera1 = ET.Element("camera")
        camera1.set("name", "tracking")
        camera1.set("pos", "1.5 -1.5 1")
        camera1.set("xyaxes", "0.707 0.707 0 -0.3 0.3 0.9")
        camera1.set("mode", "trackcom")
        root_body.insert(1, camera1)

        # Add freejoint for floating base
        freejoint = ET.Element("freejoint")
        freejoint.set("name", "floating_base_joint")
        # Insert after inertial if it exists
        insert_idx = 2
        for i, child in enumerate(root_body):
            if child.tag == "inertial":
                insert_idx = i + 1
                break
        root_body.insert(insert_idx, freejoint)

        worldbody.append(root_body)

        # Add sites and sensors
        add_sites_and_sensors(
            root_body, worldbody, mujoco,
            imu_site_name=imu_site_name,
            contact_body_names=contact_body_names,
            sensors=sensors
        )

    # Extract and add actuators if requested
    if include_actuators:
        actuator_list = []
        for body in worldbody.findall(".//body"):
            extract_actuators(body, actuator_list)

        # Add actuators
        if actuator_list:
            actuator_section = ET.SubElement(mujoco, "actuator")
            for joint_name, effort, velocity in actuator_list:
                motor = ET.SubElement(actuator_section, "motor")
                motor.set("name", f"{joint_name}_motor")
                motor.set("joint", joint_name)
                # motor.set("gear", str(effort))
                motor.set("ctrllimited", "true")
                motor.set("ctrlrange", f"-{effort} {effort}")
    else:
        # Remove temporary effort/velocity attributes from joints
        for joint in worldbody.findall(".//joint"):
            if "_effort" in joint.attrib:
                del joint.attrib["_effort"]
            if "_velocity" in joint.attrib:
                del joint.attrib["_velocity"]

    # Pretty print with indentation
    def indent_xml(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent_xml(mujoco)

    # Write to file
    tree = ET.ElementTree(mujoco)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"MuJoCo XML written to: {output_path}")
