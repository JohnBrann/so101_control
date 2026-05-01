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
    hardware_type       = LaunchConfiguration("hardware_type")
    follower_ns         = LaunchConfiguration("follower_namespace")
    follower_frame_prefix = LaunchConfiguration("follower_frame_prefix")
    follower_usb        = LaunchConfiguration("follower_usb_port")
    follower_joint_cfg  = LaunchConfiguration("follower_joint_config_file")
    follower_ctrl_cfg   = LaunchConfiguration("follower_controller_config_file")
    follower_rviz       = LaunchConfiguration("follower_rviz")

    use_cameras         = LaunchConfiguration("use_cameras")
    cameras_config_file = LaunchConfiguration("cameras_config_file")
    use_camera_tf       = LaunchConfiguration("use_camera_tf")
    use_teleop_rviz     = LaunchConfiguration("use_teleop_rviz")

    checkpoint_path     = LaunchConfiguration("checkpoint_path")
    obs_dim             = LaunchConfiguration("obs_dim")
    act_dim             = LaunchConfiguration("act_dim")
    rl_delay_s          = LaunchConfiguration("rl_delay_s")

    # --- Follower bringup only (no leader) ---
    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "follower.launch.py"])
        ),
        launch_arguments={
            "namespace":              follower_ns,
            "hardware_type":          hardware_type,
            "usb_port":               follower_usb,
            "frame_prefix":           follower_frame_prefix,
            "joint_config_file":      follower_joint_cfg,
            "controller_config_file": follower_ctrl_cfg,
            "use_rviz":               follower_rviz,
            "arm_controller":         "forward_controller",
        }.items(),
    )

    # --- TF launches (keep everything visible in RViz) ---
    layout_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "layout_tf.launch.py"])
        ),
    )

    camera_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "camera_tf.launch.py"])
        ),
        condition=IfCondition(use_camera_tf),
    )

    # --- Cameras ---
    cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "cameras.launch.py"])
        ),
        condition=IfCondition(use_cameras),
        launch_arguments={
            "cameras_config": cameras_config_file,
        }.items(),
    )

    # --- RViz ---
    teleop_rviz = PathJoinSubstitution([FindPackageShare("so101_bringup"), "rviz", "teleop.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="teleop_rviz",
        arguments=["-d", teleop_rviz],
        condition=IfCondition(use_teleop_rviz),
        output="screen",
    )

    # --- Object pose publisher (april tag → /object_pose) ---
    object_pose_node = Node(
        package="so101_control",
        executable="object_pose_estimation",
        name="object_pose_estimation",
        output="screen",
    )

    # --- RL control node — delayed so joints/TF are ready first ---
    rl_control_node = Node(
        package="so101_control",
        executable="rl_control",
        name="rl_control_node",
        output="screen",
        parameters=[{
            "checkpoint_path": checkpoint_path,
            "obs_dim":         obs_dim,
            "act_dim":         act_dim,
        }],
    )

    rl_control_start = TimerAction(
        period=rl_delay_s,
        actions=[rl_control_node],
    )

    # --- Defaults ---
    default_follower_ctrl_cfg = PathJoinSubstitution([
        FindPackageShare("so101_bringup"),
        "config", "ros2_control", "follower_controllers.yaml",
    ])
    default_cameras_cfg = PathJoinSubstitution([
        FindPackageShare("so101_bringup"),
        "config", "cameras", "so101_cameras.yaml",
    ])

    return LaunchDescription([
        # Hardware
        DeclareLaunchArgument("hardware_type",                default_value="real"),
        DeclareLaunchArgument("follower_namespace",           default_value="follower"),
        DeclareLaunchArgument("follower_frame_prefix",        default_value="follower/"),
        DeclareLaunchArgument("follower_usb_port",            default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("follower_joint_config_file",   default_value=""),
        DeclareLaunchArgument("follower_controller_config_file", default_value=default_follower_ctrl_cfg),
        DeclareLaunchArgument("follower_rviz",                default_value="false"),

        # Cameras / TF
        DeclareLaunchArgument("use_cameras",                  default_value="true"),
        DeclareLaunchArgument("cameras_config_file",          default_value=default_cameras_cfg),
        DeclareLaunchArgument("use_camera_tf",                default_value="true"),
        DeclareLaunchArgument("use_teleop_rviz",              default_value="true"),

        # RL node
        DeclareLaunchArgument("checkpoint_path",              default_value=""),
        DeclareLaunchArgument("obs_dim",                      default_value="25"),
        DeclareLaunchArgument("act_dim",                      default_value="6"),
        DeclareLaunchArgument("rl_delay_s",                   default_value="5.0"),

        # Actions
        follower_launch,
        layout_tf_launch,
        camera_tf_launch,
        cameras_launch,
        rviz_node,
        object_pose_node,
        rl_control_start,
    ])