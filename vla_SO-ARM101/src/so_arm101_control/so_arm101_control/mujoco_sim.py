#!/usr/bin/env python3
"""MuJoCo simulation bridge node for SO-ARM101.

Replaces gz_ros2_control + ros_gz_bridge. Mirrors joint states from
ros2_control (mock hardware) into MuJoCo for visualization and camera
rendering. Does NOT drive the control loop — ros2_control + MoveIt
work identically to mock hardware mode.

Responsibilities:
  1. Load URDF + scene MJCF, compile MuJoCo model
  2. Step physics at 1kHz, publish /clock
  3. Subscribe to /joint_states, update MuJoCo qpos
  4. Render wrist camera at 30Hz, publish /wrist_camera + /camera_info
  5. Subscribe to /mujoco/set_body_pose for object repositioning
  6. Optionally launch MuJoCo viewer window

Usage (via launch file):
  ros2 launch so_arm101_control mujoco.launch.py
"""

import math
import os
import re
import threading
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState


def resolve_package_uris(xml_string):
    """Replace package:// URIs with absolute filesystem paths."""
    def _replacer(match):
        pkg_name = match.group(1)
        rel_path = match.group(2)
        try:
            pkg_dir = get_package_share_directory(pkg_name)
            return os.path.join(pkg_dir, rel_path)
        except Exception:
            return match.group(0)

    return re.sub(r'package://([^/]+)/(.*?)(?=["\s<>])', _replacer, xml_string)


def collect_mesh_assets(xml_string):
    """Collect all mesh files referenced in XML as {basename: bytes} dict.

    MuJoCo's URDF parser strips directories from mesh filenames and looks
    them up by basename in the assets dict. We read the actual files
    (after package:// resolution) and provide them as binary assets.
    """
    assets = {}
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_string, re.IGNORECASE):
        filepath = match.group(1)
        basename = os.path.basename(filepath)
        if basename not in assets and os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                assets[basename] = f.read()
    return assets


def load_urdf_model(urdf_xml):
    """Compile URDF string into MuJoCo model, handling mesh assets."""
    assets = collect_mesh_assets(urdf_xml)
    return mujoco.MjModel.from_xml_string(urdf_xml, assets=assets)


def merge_urdf_into_scene(urdf_xml, scene_file):
    """Merge URDF robot model into MuJoCo scene MJCF.

    Strategy: Convert URDF to MJCF via mujoco, then inject the robot
    body tree into the scene's worldbody. Reuse the same mesh assets
    dict throughout (mj_saveLastXML writes basenames only when assets
    were loaded via dict, so we can't re-collect from disk).
    """
    import tempfile

    # Collect mesh assets from resolved URDF (absolute paths on disk)
    mesh_assets = collect_mesh_assets(urdf_xml)

    # Compile URDF standalone to get MJCF representation
    urdf_model = mujoco.MjModel.from_xml_string(urdf_xml, assets=mesh_assets)

    # Save URDF-compiled model to temporary MJCF
    with tempfile.NamedTemporaryFile(suffix='.xml', mode='w', delete=False) as f:
        tmp_path = f.name
    mujoco.mj_saveLastXML(tmp_path, urdf_model)

    # Parse both XMLs
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

    # Ensure offscreen framebuffer is large enough for camera rendering
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

    # Merge contact/equality/tendon if present
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
    return merged_xml, mesh_assets


class MujocoSimNode(Node):
    """MuJoCo simulation bridge for SO-ARM101."""

    # Joint names matching ros2_control config
    JOINT_NAMES = [
        'shoulder_pan', 'shoulder_lift', 'elbow_flex',
        'wrist_flex', 'wrist_roll', 'gripper_joint',
    ]

    def __init__(self):
        super().__init__('mujoco_sim')

        # Parameters
        self.declare_parameter('scene_file', '')
        self.declare_parameter('headless', False)
        self.declare_parameter('camera_width', 1280)
        self.declare_parameter('camera_height', 720)
        self.declare_parameter('camera_hfov', 1.7453)  # 100 degrees in radians
        self.declare_parameter('camera_fps', 30.0)
        self.declare_parameter('physics_rate', 1000.0)

        scene_file = self.get_parameter('scene_file').value
        self.headless = self.get_parameter('headless').value
        self.cam_width = self.get_parameter('camera_width').value
        self.cam_height = self.get_parameter('camera_height').value
        self.cam_hfov = self.get_parameter('camera_hfov').value
        self.cam_fps = self.get_parameter('camera_fps').value
        physics_rate = self.get_parameter('physics_rate').value

        if not scene_file:
            desc_pkg = get_package_share_directory('so_arm101_description')
            scene_file = os.path.join(desc_pkg, 'mujoco', 'so_arm101_scene.xml')

        # Get URDF from robot_description parameter (published by robot_state_publisher)
        self.declare_parameter('robot_description', '')
        urdf_string = self.get_parameter('robot_description').value

        if not urdf_string:
            # Try to get from /robot_description topic
            self.get_logger().info('No robot_description parameter, subscribing to topic...')
            self._urdf_received = threading.Event()
            self._urdf_string = None
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(
                std_msgs_String, '/robot_description', self._robot_desc_cb, qos)
            if not self._urdf_received.wait(timeout=10.0):
                self.get_logger().error('Timeout waiting for robot_description')
                raise RuntimeError('No robot_description available')
            urdf_string = self._urdf_string

        # Resolve package:// URIs to absolute paths
        urdf_string = resolve_package_uris(urdf_string)

        # Load model
        self.get_logger().info(f'Loading MuJoCo scene: {scene_file}')
        try:
            merged_xml, mesh_assets = merge_urdf_into_scene(urdf_string, scene_file)
            self.model = mujoco.MjModel.from_xml_string(merged_xml, assets=mesh_assets)
        except Exception as e:
            self.get_logger().warn(f'Merged model failed ({e}), loading URDF standalone...')
            self.model = load_urdf_model(urdf_string)

        self.data = mujoco.MjData(self.model)
        self.get_logger().info(
            f'MuJoCo model loaded: {self.model.nq} qpos, {self.model.nv} qvel, '
            f'{self.model.nbody} bodies, {self.model.njnt} joints')

        # Build joint name -> qpos index mapping
        self.joint_map = {}
        for name in self.JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.joint_map[name] = qpos_adr
                self.get_logger().info(f'  Joint {name}: id={jnt_id}, qpos_adr={qpos_adr}')
            else:
                self.get_logger().warn(f'  Joint {name}: NOT FOUND in MuJoCo model')

        # Lock for thread-safe data access
        self._lock = threading.Lock()

        # Publishers
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.image_pub = self.create_publisher(Image, '/wrist_camera', 10)
        self.caminfo_pub = self.create_publisher(CameraInfo, '/camera_info', 10)

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.create_subscription(
            PoseStamped, '/mujoco/set_body_pose', self._set_body_pose_cb, 10)

        # Compute camera intrinsics
        self._compute_camera_intrinsics()

        # Renderer for camera images
        self._renderer = None
        self._init_renderer()

        # Physics timer (1kHz)
        physics_period = 1.0 / physics_rate
        self.create_timer(physics_period, self._physics_step)

        # Camera render timer (30Hz)
        cam_period = 1.0 / self.cam_fps
        self.create_timer(cam_period, self._render_camera)

        # Launch viewer in separate thread if not headless
        self._viewer_handle = None
        if not self.headless:
            self._start_viewer()

        self.get_logger().info('MuJoCo simulation node ready')

    def _compute_camera_intrinsics(self):
        """Compute camera intrinsic matrix from hfov and resolution."""
        fx = self.cam_width / (2.0 * math.tan(self.cam_hfov / 2.0))
        fy = fx  # Square pixels
        cx = self.cam_width / 2.0
        cy = self.cam_height / 2.0
        self._K = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

    def _init_renderer(self):
        """Initialize MuJoCo renderer for camera images."""
        try:
            self._renderer = mujoco.Renderer(
                self.model, height=self.cam_height, width=self.cam_width)
            self.get_logger().info(
                f'Renderer initialized: {self.cam_width}x{self.cam_height}')
        except Exception as e:
            self.get_logger().warn(f'Could not initialize renderer: {e}')
            self._renderer = None

    def _start_viewer(self):
        """Start MuJoCo interactive viewer in a background thread."""
        def _run_viewer():
            try:
                with mujoco.viewer.launch_passive(
                    self.model, self.data, key_callback=None
                ) as viewer:
                    self._viewer_handle = viewer
                    while viewer.is_running():
                        with self._lock:
                            viewer.sync()
                        import time
                        time.sleep(0.02)  # ~50fps viewer update
            except Exception as e:
                self.get_logger().warn(f'Viewer error: {e}')

        viewer_thread = threading.Thread(target=_run_viewer, daemon=True)
        viewer_thread.start()
        self.get_logger().info('MuJoCo viewer launched')

    def _joint_state_cb(self, msg):
        """Update MuJoCo qpos from /joint_states."""
        with self._lock:
            changed = False
            for name, position in zip(msg.name, msg.position):
                if name in self.joint_map:
                    idx = self.joint_map[name]
                    if abs(self.data.qpos[idx] - position) > 1e-6:
                        changed = True
                    self.data.qpos[idx] = position
            if changed:
                self.get_logger().info(
                    f'Joint update: {dict(zip(msg.name, [f"{p:.4f}" for p in msg.position]))}',
                    throttle_duration_sec=1.0)

    def _set_body_pose_cb(self, msg):
        """Reposition a mocap body via /mujoco/set_body_pose.

        Body name is in header.frame_id.
        """
        body_name = msg.header.frame_id
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            self.get_logger().warn(f'Body "{body_name}" not found in MuJoCo model')
            return

        # Find mocap body index
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            self.get_logger().warn(f'Body "{body_name}" is not a mocap body')
            return

        p = msg.pose.position
        q = msg.pose.orientation

        with self._lock:
            self.data.mocap_pos[mocap_id] = [p.x, p.y, p.z]
            self.data.mocap_quat[mocap_id] = [q.w, q.x, q.y, q.z]

    def _physics_step(self):
        """Update MuJoCo kinematics and publish /clock.

        Uses mj_forward() instead of mj_step() because we mirror joint
        positions from ros2_control mock hardware. mj_step() would run
        physics that overwrites our qpos; mj_forward() just computes
        kinematics/rendering from the current qpos.
        """
        with self._lock:
            mujoco.mj_forward(self.model, self.data)
            self.data.time += self.model.opt.timestep
            sim_time = self.data.time

        # Publish clock
        clock_msg = Clock()
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        clock_msg.clock = Time(sec=sec, nanosec=nanosec)
        self.clock_pub.publish(clock_msg)

    def _render_camera(self):
        """Render wrist camera and publish image + camera_info."""
        if self._renderer is None:
            return

        # Find camera_link body to position the MuJoCo camera
        cam_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'camera_link')

        try:
            with self._lock:
                # Update scene
                self._renderer.update_scene(self.data)

                if cam_body_id >= 0:
                    # Position camera at camera_link body
                    cam_pos = self.data.xpos[cam_body_id].copy()
                    cam_mat = self.data.xmat[cam_body_id].reshape(3, 3).copy()

                    # camera_link frame: X=forward, Y=left, Z=up (ROS convention)
                    # MuJoCo camera: -Z=forward, X=right, Y=up
                    # So we need: mj_forward = link_X, mj_up = link_Z
                    lookat = cam_pos + cam_mat[:, 0] * 0.3  # Look along X axis

                    self._renderer.update_scene(
                        self.data,
                        camera=mujoco.MjvCamera(
                            type=mujoco.mjtCamera.mjCAMERA_FREE,
                            lookat=lookat,
                            distance=0.3,
                            azimuth=0,
                            elevation=0,
                        ),
                    )

                # Render
                img = self._renderer.render()

            # Build timestamp from sim time
            with self._lock:
                sim_time = self.data.time
            sec = int(sim_time)
            nanosec = int((sim_time - sec) * 1e9)
            stamp = Time(sec=sec, nanosec=nanosec)

            # Publish Image
            img_msg = Image()
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = 'camera_optical_frame'
            img_msg.height = self.cam_height
            img_msg.width = self.cam_width
            img_msg.encoding = 'rgb8'
            img_msg.is_bigendian = False
            img_msg.step = self.cam_width * 3
            img_msg.data = img.tobytes()
            self.image_pub.publish(img_msg)

            # Publish CameraInfo
            info_msg = CameraInfo()
            info_msg.header.stamp = stamp
            info_msg.header.frame_id = 'camera_optical_frame'
            info_msg.height = self.cam_height
            info_msg.width = self.cam_width
            info_msg.distortion_model = 'plumb_bob'
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            info_msg.k = self._K
            info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info_msg.p = [
                self._K[0], 0.0, self._K[2], 0.0,
                0.0, self._K[4], self._K[5], 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            self.caminfo_pub.publish(info_msg)

        except Exception as e:
            self.get_logger().warn(f'Camera render error: {e}', throttle_duration_sec=5.0)

    def destroy_node(self):
        """Clean up renderer and viewer."""
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer_handle is not None:
            self._viewer_handle.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
