#!/usr/bin/env python3
"""Occlusion detection for multi-block pick-and-place.

Determines whether one block occludes another from the wrist camera's viewpoint
using simplified 2D projection. The camera is on the robot's wrist, so occlusion
depends on the arm configuration.

Usage:
    from so_arm101_control.occlusion import is_occluded
    occluded = is_occluded(
        target_pos=(0.18, 0.03),
        target_half_size=(0.016, 0.008),
        occluder_pos=(0.19, 0.035),
        occluder_half_size=(0.008, 0.008),
        camera_pos=np.array([0.15, 0.0, 0.2]),
    )
"""

import numpy as np

# Rotation from camera_link frame to OpenCV frame (from randomize_legos.py)
LINK_TO_OPENCV = np.array([
    [0, -1, 0],
    [0,  0, -1],
    [1,  0, 0],
], dtype=float)

# Approximate camera intrinsics (from mujoco_sim.py: hfov=100°, 1280x720)
# fx = width / (2 * tan(hfov/2)) = 1280 / (2 * tan(50°)) ≈ 537
_FX = 537.0
_FY = 537.0
_CX = 640.0
_CY = 360.0


def _project_to_camera(world_pos, camera_pos, camera_rot):
    """Project a 3D world point to 2D camera pixel coordinates.

    Args:
        world_pos: (3,) world position.
        camera_pos: (3,) camera position in world.
        camera_rot: (3, 3) rotation matrix of camera_link in world frame.

    Returns:
        (u, v, depth) or None if behind camera.
    """
    v_world = world_pos - camera_pos
    v_link = camera_rot.T @ v_world
    v_cv = LINK_TO_OPENCV @ v_link

    if v_cv[2] <= 0:
        return None

    u = _FX * v_cv[0] / v_cv[2] + _CX
    v = _FY * v_cv[1] / v_cv[2] + _CY
    return u, v, v_cv[2]


def _project_block_bbox(block_xy, block_half_size, block_yaw, table_z,
                        camera_pos, camera_rot):
    """Project a block's 4 corners to camera pixel coordinates.

    Returns:
        (u_min, v_min, u_max, v_max, depth) or None if behind camera.
    """
    hl, hw = block_half_size
    cy, sy = np.cos(block_yaw), np.sin(block_yaw)

    # 4 corners of the block on the table
    corners_local = np.array([
        [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]
    ])

    rot_2d = np.array([[cy, -sy], [sy, cy]])
    corners_world = (rot_2d @ corners_local.T).T + block_xy

    us, vs = [], []
    depths = []
    for cx, cy_val in corners_world:
        result = _project_to_camera(
            np.array([cx, cy_val, table_z]),
            camera_pos, camera_rot
        )
        if result is None:
            return None
        u, v, d = result
        us.append(u)
        vs.append(v)
        depths.append(d)

    return min(us), min(vs), max(us), max(vs), np.mean(depths)


def _bboxes_overlap(bbox1, bbox2):
    """Check if two 2D bounding boxes overlap.

    Each bbox is (u_min, v_min, u_max, v_max).
    """
    return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])


def is_occluded(target_pos, target_half_size, target_yaw,
                occluder_pos, occluder_half_size, occluder_yaw,
                camera_pos, camera_rot, table_z=0.0055):
    """Check if target block is occluded by occluder block from camera viewpoint.

    Args:
        target_pos: (x, y) of target block center.
        target_half_size: (half_length, half_width) of target.
        target_yaw: Yaw angle of target block.
        occluder_pos: (x, y) of occluder block center.
        occluder_half_size: (half_length, half_width) of occluder.
        occluder_yaw: Yaw angle of occluder block.
        camera_pos: (3,) camera position in world.
        camera_rot: (3, 3) rotation matrix of camera_link body.
        table_z: Height of table surface.

    Returns:
        True if target is occluded by occluder.
    """
    target_proj = _project_block_bbox(
        np.asarray(target_pos), target_half_size, target_yaw, table_z,
        camera_pos, camera_rot
    )
    occluder_proj = _project_block_bbox(
        np.asarray(occluder_pos), occluder_half_size, occluder_yaw, table_z,
        camera_pos, camera_rot
    )

    if target_proj is None or occluder_proj is None:
        return False

    target_bbox = target_proj[:4]
    target_depth = target_proj[4]
    occluder_bbox = occluder_proj[:4]
    occluder_depth = occluder_proj[4]

    # Occluder must be closer to camera than target
    if occluder_depth >= target_depth:
        return False

    return _bboxes_overlap(target_bbox, occluder_bbox)


def _point_in_rotated_rect(point_xy, rect_center, rect_half_size, rect_yaw):
    """Check if a 2D point falls inside a rotated rectangle.

    Args:
        point_xy: (x, y) point to test.
        rect_center: (x, y) center of rectangle.
        rect_half_size: (half_length, half_width) of rectangle.
        rect_yaw: Yaw angle of rectangle (radians).

    Returns:
        True if point is inside the rotated rectangle.
    """
    dx = point_xy[0] - rect_center[0]
    dy = point_xy[1] - rect_center[1]
    cy, sy = np.cos(rect_yaw), np.sin(rect_yaw)
    local_x = dx * cy + dy * sy
    local_y = -dx * sy + dy * cy
    hl, hw = rect_half_size
    return abs(local_x) <= hl and abs(local_y) <= hw


def is_occluded_overhead(target_pos, occluder_pos, occluder_half_size, occluder_yaw):
    """Check if target block center is occluded by occluder from directly overhead.

    Uses top-down shadow model: occluder's 2D footprint shadows the target center.

    Args:
        target_pos: (x, y) of target block center.
        occluder_pos: (x, y) of occluder block center.
        occluder_half_size: (half_length, half_width) of occluder.
        occluder_yaw: Yaw angle of occluder block (radians).

    Returns:
        True if target center falls inside occluder's XY footprint.
    """
    return _point_in_rotated_rect(target_pos, occluder_pos,
                                   occluder_half_size, occluder_yaw)
