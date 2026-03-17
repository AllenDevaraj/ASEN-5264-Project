#!/usr/bin/env python3
"""Standalone MuJoCo model loader for SO-ARM101 (no ROS2 dependency).

Replicates the URDF-to-MJCF merge pipeline from mujoco_sim.py but reads
files directly from disk instead of ROS2 topics/parameters.

Usage:
    from so_arm101_control.model_loader import load_mujoco_model
    model, data = load_mujoco_model()
"""

import os
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# Default paths (relative to this file's package)
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DESC_PKG = os.path.join(_PKG_ROOT, 'so_arm101_description')

DEFAULT_URDF = os.path.join(_DESC_PKG, 'urdf', 'so_arm101.urdf')
DEFAULT_SCENE = os.path.join(_DESC_PKG, 'mujoco', 'so_arm101_scene.xml')
DEFAULT_MESHES = os.path.join(_DESC_PKG, 'meshes')


def _resolve_package_uris(xml_string, package_roots):
    """Replace package:// URIs with absolute filesystem paths.

    Args:
        xml_string: XML content with package:// references.
        package_roots: dict mapping package_name -> absolute path.
    """
    def _replacer(match):
        pkg_name = match.group(1)
        rel_path = match.group(2)
        if pkg_name in package_roots:
            return os.path.join(package_roots[pkg_name], rel_path)
        return match.group(0)

    return re.sub(r'package://([^/]+)/(.*?)(?=["\s<>])', _replacer, xml_string)


def _collect_mesh_assets(xml_string):
    """Collect all mesh files referenced in XML as {basename: bytes} dict."""
    assets = {}
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_string, re.IGNORECASE):
        filepath = match.group(1)
        basename = os.path.basename(filepath)
        if basename not in assets and os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                assets[basename] = f.read()
    return assets


def _merge_urdf_into_scene(urdf_xml, scene_file, mesh_assets):
    """Merge URDF robot model into MuJoCo scene MJCF.

    Adapted from mujoco_sim.py:merge_urdf_into_scene.
    """
    # Compile URDF standalone to get MJCF representation
    urdf_model = mujoco.MjModel.from_xml_string(urdf_xml, assets=mesh_assets)

    with tempfile.NamedTemporaryFile(suffix='.xml', mode='w', delete=False) as f:
        tmp_path = f.name
    mujoco.mj_saveLastXML(tmp_path, urdf_model)

    scene_root = ET.parse(scene_file).getroot()
    robot_root = ET.parse(tmp_path).getroot()
    os.unlink(tmp_path)

    # Strip robot's compiler element (avoid conflicts with scene)
    robot_compiler = robot_root.find('compiler')
    if robot_compiler is not None:
        robot_root.remove(robot_compiler)

    # Strip scene's compiler meshdir (we use assets dict)
    scene_compiler = scene_root.find('compiler')
    if scene_compiler is not None:
        if 'meshdir' in scene_compiler.attrib:
            del scene_compiler.attrib['meshdir']

    # Ensure offscreen framebuffer for rendering
    scene_visual = scene_root.find('visual')
    if scene_visual is None:
        scene_visual = ET.SubElement(scene_root, 'visual')
    vis_global = scene_visual.find('global')
    if vis_global is None:
        vis_global = ET.SubElement(scene_visual, 'global')
    vis_global.set('offwidth', '1280')
    vis_global.set('offheight', '720')

    # Merge assets
    scene_asset = scene_root.find('asset')
    robot_asset = robot_root.find('asset')
    if robot_asset is not None:
        if scene_asset is None:
            scene_root.append(robot_asset)
        else:
            for elem in robot_asset:
                scene_asset.append(elem)

    # Merge defaults
    scene_default = scene_root.find('default')
    robot_default = robot_root.find('default')
    if robot_default is not None:
        if scene_default is None:
            scene_root.insert(0, robot_default)
        else:
            for elem in robot_default:
                scene_default.append(elem)

    # Add robot body to worldbody
    scene_worldbody = scene_root.find('worldbody')
    robot_worldbody = robot_root.find('worldbody')
    if robot_worldbody is not None:
        for body in robot_worldbody:
            scene_worldbody.append(body)

    # Merge actuators
    scene_actuator = scene_root.find('actuator')
    robot_actuator = robot_root.find('actuator')
    if robot_actuator is not None:
        if scene_actuator is None:
            scene_root.append(robot_actuator)
        else:
            for elem in robot_actuator:
                scene_actuator.append(elem)

    # Merge contact/equality/tendon/sensor
    for tag in ['contact', 'equality', 'tendon', 'sensor']:
        robot_elem = robot_root.find(tag)
        scene_elem = scene_root.find(tag)
        if robot_elem is not None:
            if scene_elem is None:
                scene_root.append(robot_elem)
            else:
                for child in robot_elem:
                    scene_elem.append(child)

    merged_xml = ET.tostring(scene_root, encoding='unicode')
    return merged_xml


def load_mujoco_model(urdf_path=None, scene_path=None, meshes_dir=None):
    """Load merged robot+scene MuJoCo model without ROS2.

    Args:
        urdf_path: Path to so_arm101.urdf. Defaults to package path.
        scene_path: Path to so_arm101_scene.xml. Defaults to package path.
        meshes_dir: Path to meshes/ directory. Defaults to package path.

    Returns:
        (mujoco.MjModel, mujoco.MjData) tuple.
    """
    urdf_path = urdf_path or DEFAULT_URDF
    scene_path = scene_path or DEFAULT_SCENE
    meshes_dir = meshes_dir or DEFAULT_MESHES

    # Build package roots for URI resolution
    package_roots = {
        'so_arm101_description': _DESC_PKG,
        'so_arm101_control': os.path.join(_PKG_ROOT, 'so_arm101_control'),
    }

    # Read and resolve URDF
    with open(urdf_path, 'r') as f:
        urdf_xml = f.read()
    urdf_xml = _resolve_package_uris(urdf_xml, package_roots)

    # Collect mesh assets
    mesh_assets = _collect_mesh_assets(urdf_xml)

    # Also collect any meshes from meshes_dir that might be missed
    if os.path.isdir(meshes_dir):
        for fname in os.listdir(meshes_dir):
            if fname.lower().endswith('.stl') and fname not in mesh_assets:
                with open(os.path.join(meshes_dir, fname), 'rb') as f:
                    mesh_assets[fname] = f.read()

    # Merge and compile
    merged_xml = _merge_urdf_into_scene(urdf_xml, scene_path, mesh_assets)
    model = mujoco.MjModel.from_xml_string(merged_xml, assets=mesh_assets)
    data = mujoco.MjData(model)

    return model, data


def build_joint_map(model):
    """Build a mapping from joint name -> qpos index.

    Returns:
        dict mapping joint name (str) to qpos index (int).
    """
    joint_map = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_map[name] = model.jnt_qposadr[i]
    return joint_map


def build_mocap_map(model):
    """Build a mapping from mocap body name -> mocap index.

    Returns:
        dict mapping body name (str) to mocap index (int).
    """
    mocap_map = {}
    for i in range(model.nbody):
        if model.body_mocapid[i] >= 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                mocap_map[name] = model.body_mocapid[i]
    return mocap_map
