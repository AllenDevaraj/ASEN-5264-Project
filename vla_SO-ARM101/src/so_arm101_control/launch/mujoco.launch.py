"""
MuJoCo simulation launch for SO-ARM101.
Replaces gazebo.launch.py — uses MuJoCo for physics visualization and camera
rendering while ros2_control runs on mock hardware (same as control.launch.py).

Usage:
  ros2 launch so_arm101_control mujoco.launch.py
  ros2 launch so_arm101_control mujoco.launch.py headless:=true rviz:=false
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg = get_package_share_directory('so_arm101_description')
    moveit_pkg = get_package_share_directory('so_arm101_moveit_config')

    # Process the MuJoCo-specific xacro (includes base URDF + mock hardware ros2_control)
    xacro_file = os.path.join(desc_pkg, 'urdf', 'so_arm101.mujoco.urdf.xacro')
    robot_description = subprocess.check_output(
        ['xacro', xacro_file]).decode('utf-8')

    controllers_yaml = os.path.join(moveit_pkg, 'config', 'ros2_controllers.yaml')
    scene_file = os.path.join(desc_pkg, 'mujoco', 'so_arm101_scene.xml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('rviz')
    headless = LaunchConfiguration('headless')

    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulated clock from MuJoCo'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Launch RViz visualization'),
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run without MuJoCo viewer window'),

        # Robot State Publisher (with xacro-processed URDF including ros2_control)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }],
            output='screen',
        ),

        # ros2_control controller manager (mock hardware)
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                {'robot_description': robot_description},
                controllers_yaml,
            ],
            output='screen',
        ),

        # Spawn controllers
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_controller'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['gripper_controller'],
            output='screen',
        ),

        # MuJoCo simulation node — uses wall clock (it IS the /clock source)
        Node(
            package='so_arm101_control',
            executable='mujoco_sim',
            name='mujoco_sim',
            parameters=[{
                'scene_file': scene_file,
                'headless': headless,
                'camera_width': 1280,
                'camera_height': 720,
                'camera_hfov': 1.7453,
                'camera_fps': 30.0,
                'robot_description': robot_description,
                'use_sim_time': False,
            }],
            output='screen',
        ),

        # MoveIt move_group
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')),
        ),

        # Control GUI
        Node(
            package='so_arm101_control',
            executable='control_gui',
            name='so_arm101_control_gui',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),

        # EE Pose Publisher
        Node(
            package='so_arm101_control',
            executable='ee_pose_publisher',
            name='ee_pose_publisher',
            parameters=[{
                'base_frame': 'base',
                'ee_frame': 'gripper',
                'publish_rate': 10.0,
                'startup_delay': 5.0,
                'use_sim_time': True,
            }],
            output='screen',
        ),

        # Camera Pose Publisher
        Node(
            package='so_arm101_control',
            executable='camera_pose_publisher',
            name='camera_pose_publisher',
            parameters=[{
                'base_frame': 'base',
                'camera_frame': 'camera_link',
                'publish_rate': 10.0,
                'startup_delay': 5.0,
                'use_sim_time': True,
            }],
            output='screen',
        ),

        # RViz (via MoveIt launch for full plugin support)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_pkg, 'launch', 'moveit_rviz.launch.py')),
            condition=IfCondition(use_rviz),
        ),
    ])
