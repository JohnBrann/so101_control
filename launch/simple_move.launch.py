from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    leader_ns = LaunchConfiguration("leader_namespace")
    follower_ns = LaunchConfiguration("follower_namespace")
    leader_frame_prefix = LaunchConfiguration("leader_frame_prefix")
    follower_frame_prefix = LaunchConfiguration("follower_frame_prefix")

    leader_usb = LaunchConfiguration("leader_usb_port")
    follower_usb = LaunchConfiguration("follower_usb_port")

    leader_joint_cfg = LaunchConfiguration("leader_joint_config_file")
    follower_joint_cfg = LaunchConfiguration("follower_joint_config_file")

    leader_ctrl_cfg = LaunchConfiguration("leader_controller_config_file")
    follower_ctrl_cfg = LaunchConfiguration("follower_controller_config_file")

    leader_rviz = LaunchConfiguration("leader_rviz")
    follower_rviz = LaunchConfiguration("follower_rviz")

    arm_controller = LaunchConfiguration("arm_controller")  # trajectory_controller|forward_controller

    move_delay_s = LaunchConfiguration("move_delay_s")

    use_cameras = LaunchConfiguration("use_cameras")
    cameras_config_file = LaunchConfiguration("cameras_config_file")
    use_camera_tf = LaunchConfiguration("use_camera_tf")

    use_move_rviz = LaunchConfiguration("use_move_rviz")

    use_rerun = LaunchConfiguration("use_rerun")
    rerun_env_dir = LaunchConfiguration("rerun_env_dir")
    rerun_delay_s = LaunchConfiguration("rerun_delay_s")

    # --- Include leader bringup ---
    leader_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "leader.launch.py"])
        ),
        launch_arguments={
            "namespace": leader_ns,
            "hardware_type": hardware_type,
            "usb_port": leader_usb,
            "frame_prefix": leader_frame_prefix,
            "joint_config_file": leader_joint_cfg,
            "controller_config_file": leader_ctrl_cfg,
            "use_rviz": leader_rviz,
        }.items(),
    )

    # --- Include follower bringup ---
    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "follower.launch.py"])
        ),
        launch_arguments={
            "namespace": follower_ns,
            "hardware_type": hardware_type,
            "usb_port": follower_usb,
            "frame_prefix": follower_frame_prefix,
            "joint_config_file": follower_joint_cfg,
            "controller_config_file": follower_ctrl_cfg,
            "use_rviz": follower_rviz,
            "arm_controller": arm_controller,
        }.items(),
    )

    # --- Start your script node instead of teleop ---
    move_node = Node(
        package="so101_control",
        executable="simple_move",
        name="simple_move",
        output="screen",
    )

    move_start = TimerAction(
        period=move_delay_s,
        actions=[move_node],
    )

    # --- Include cameras launch ---
    cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "cameras.launch.py"])
        ),
        condition=IfCondition(use_cameras),
        launch_arguments={
            "cameras_config": cameras_config_file,
        }.items(),
    )

    camera_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "camera_tf.launch.py"])
        ),
        condition=IfCondition(use_camera_tf),
        launch_arguments={}.items(),
    )

    # --- Include layout launch tf ---
    layout_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "layout_tf.launch.py"])
        ),
    )

    # --- RViz node ---
    move_rviz = PathJoinSubstitution([FindPackageShare("so101_bringup"), "rviz", "teleop.rviz"])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="move_rviz",
        arguments=["-d", move_rviz],
        condition=IfCondition(use_move_rviz),
        output="screen",
    )

    # --- Launch Rerun ---
    rerun_bridge_proc = ExecuteProcess(
        cmd=[
            "pixi",
            "run",
            "bridge",
            "--",
        ],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        condition=IfCondition(use_rerun),
        output="screen",
    )

    rerun_start = TimerAction(
        period=rerun_delay_s,
        actions=[rerun_bridge_proc],
    )

    # --- Defaults for files ---
    default_leader_joint_cfg = ""
    default_follower_joint_cfg = ""

    default_leader_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "leader_controllers.yaml",
        ]
    )

    default_follower_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "follower_controllers.yaml",
        ]
    )

    default_cameras_cfg = PathJoinSubstitution(
        [FindPackageShare("so101_bringup"), "config", "cameras", "so101_cameras.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            DeclareLaunchArgument("leader_namespace", default_value="leader"),
            DeclareLaunchArgument("follower_namespace", default_value="follower"),
            DeclareLaunchArgument("leader_frame_prefix", default_value="leader/"),
            DeclareLaunchArgument("follower_frame_prefix", default_value="follower/"),
            DeclareLaunchArgument("leader_usb_port", default_value="/dev/ttyACM1"),
            DeclareLaunchArgument("follower_usb_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("leader_joint_config_file", default_value=default_leader_joint_cfg),
            DeclareLaunchArgument("follower_joint_config_file", default_value=default_follower_joint_cfg),
            DeclareLaunchArgument("leader_controller_config_file", default_value=default_leader_ctrl_cfg),
            DeclareLaunchArgument(
                "follower_controller_config_file",
                default_value=default_follower_ctrl_cfg,
            ),
            DeclareLaunchArgument("leader_rviz", default_value="false"),
            DeclareLaunchArgument("follower_rviz", default_value="false"),
            DeclareLaunchArgument("arm_controller", default_value="forward_controller"),
            DeclareLaunchArgument("move_delay_s", default_value="2.0"),
            DeclareLaunchArgument("use_cameras", default_value="true"),
            DeclareLaunchArgument("cameras_config_file", default_value=default_cameras_cfg),
            DeclareLaunchArgument("use_camera_tf", default_value="true"),
            DeclareLaunchArgument("use_move_rviz", default_value="true"),
            DeclareLaunchArgument("use_rerun", default_value="false"),
            DeclareLaunchArgument(
                "rerun_env_dir",
                default_value=EnvironmentVariable("SO101_RERUN_ENV_DIR", default_value=""),
            ),
            DeclareLaunchArgument("rerun_delay_s", default_value=move_delay_s),
            leader_launch,
            follower_launch,
            layout_tf_launch,
            cameras_launch,
            camera_tf_launch,
            rviz_node,
            rerun_start,
            move_start,
        ]
    )