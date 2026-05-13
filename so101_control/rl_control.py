import os
import csv
import time
import random

import numpy as np
import torch
import yaml
import rclpy

from rclpy.node import Node
from packaging import version

from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import tf2_ros
import tf2_geometry_msgs

from gymnasium.spaces import Box

import skrl
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
from skrl.resources.preprocessors.torch import RunningStandardScaler


# =============================================================================
# HELPERS
# =============================================================================

def _load_skrl_cfg(path: str) -> dict:
    """Load and return the skrl config block from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw["skrl"]


def _check_skrl_version(required: str) -> None:
    if version.parse(skrl.__version__) < version.parse(required):
        raise RuntimeError(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using: pip install skrl>={required}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# NODE
# =============================================================================

class RLControlNode(Node):

    def __init__(self):
        super().__init__("rl_control_node")

        # LOAD CONFIG FILES
        # ------------------------------------------------------------------ #

        RL_CONFIG_PATH   = "/home/csrobot/ros2_so101_ws/src/so101-ros-physical-ai/so101_control/config/rl_control.yaml"
        SKRL_CONFIG_PATH = "/home/csrobot/ros2_so101_ws/src/so101-ros-physical-ai/so101_control/config/skrl_cfg.yaml"

        with open(RL_CONFIG_PATH) as f:
            _rc = yaml.safe_load(f)["rl_control_node"]["ros__parameters"]

        with open(SKRL_CONFIG_PATH) as f:
            _skrl = yaml.safe_load(f)["skrl"]

        # ------------------------------------------------------------------ #
        # ROBOT / NODE PARAMETERS
        # ------------------------------------------------------------------ #
        self.declare_parameter("checkpoint_path", "")
        self.checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value

        # self.checkpoint_path     = _rc["checkpoint_path"]
        self.obs_dim             = _rc["obs_dim"]
        self.act_dim             = _rc["act_dim"]
        self.action_scale        = _rc["action_scale"]
        self.control_hz          = _rc["control_hz"]
        self.max_delta           = _rc["max_delta_per_step"]
        self.max_abs_action      = _rc["max_abs_action"]
        self.base_frame          = _rc["base_frame"]
        self.camera_frame        = _rc["camera_frame"]
        self.normalize_joint_pos = _rc["normalize_joint_pos"]
        self.ee_log_path         = _rc["ee_log_path"]
        self.pose_command        = list(_rc["pose_command"])
        self.joint_names         = list(_rc["joint_names"])
        self.default_joint_pos   = np.array(_rc["default_joint_pos"], dtype=np.float32)
        self.joint_min           = np.array(_rc["joint_min"],          dtype=np.float32)
        self.joint_max           = np.array(_rc["joint_max"],          dtype=np.float32)
        self.joint_range         = self.joint_max - self.joint_min

        # ------------------------------------------------------------------ #
        # SKRL CONFIG
        # ------------------------------------------------------------------ #

        _check_skrl_version(_skrl["version"])
        _seed_everything(_skrl["seed"])

        models_cfg = _skrl["model"]
        # agent_cfg kept for reference / future trainer use
        # agent_cfg = _skrl["agent"]

        # ------------------------------------------------------------------ #
        # VALIDATE CHECKPOINT
        # ------------------------------------------------------------------ #

        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            self.get_logger().error(f"Checkpoint not found: {self.checkpoint_path}")
            raise FileNotFoundError(self.checkpoint_path)

        # ------------------------------------------------------------------ #
        # DEVICE
        # ------------------------------------------------------------------ #

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # ------------------------------------------------------------------ #
        # LOAD CHECKPOINT
        # ------------------------------------------------------------------ #

        self.ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.get_logger().info(f"Checkpoint loaded from: {self.checkpoint_path}")
        self.get_logger().info(f"Checkpoint keys: {list(self.ckpt.keys())}")

        # ------------------------------------------------------------------ #
        # BUILD OBSERVATION / ACTION SPACES
        # ------------------------------------------------------------------ #

        observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        action_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.act_dim,), dtype=np.float32
        )

        # ------------------------------------------------------------------ #
        # BUILD & LOAD POLICY
        # Trained with separate=false so we must use shared_model to match
        # the checkpoint architecture (policy_layer + value_layer output heads
        # on a shared trunk) — same as what the SKRL Runner builds internally.
        # ------------------------------------------------------------------ #

        self.policy = shared_model(
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
            roles=["policy", "value"],
            parameters=[
                models_cfg["policy"],
                models_cfg["value"],
            ],
        )

        policy_state_dict = (
            self.ckpt["policy"]
            if "policy" in self.ckpt
            else {k.replace("policy.", ""): v
                  for k, v in self.ckpt.items()
                  if k.startswith("policy.")}
        )

        missing, unexpected = self.policy.load_state_dict(
            policy_state_dict, strict=True
        )
        self.get_logger().info(f"Policy missing keys:    {missing}")
        self.get_logger().info(f"Policy unexpected keys: {unexpected}")
        self.policy.eval()

        # ------------------------------------------------------------------ #
        # OBSERVATION PREPROCESSOR  (RunningStandardScaler)
        # ------------------------------------------------------------------ #

        self.obs_scaler = RunningStandardScaler(size=self.obs_dim, device=self.device)
        self.use_scaler = False

        try:
            scaler_state_dict = (
                self.ckpt["state_preprocessor"]
                if "state_preprocessor" in self.ckpt
                else {k.replace("state_preprocessor.", ""): v
                      for k, v in self.ckpt.items()
                      if k.startswith("state_preprocessor.")}
            )
            self.obs_scaler.load_state_dict(scaler_state_dict)
            self.use_scaler = True
            self.get_logger().info("RunningStandardScaler loaded successfully")

        except Exception as e:
            self.get_logger().warn(f"Failed to load scaler: {e}")
            self.get_logger().warn("Proceeding WITHOUT observation normalization")

        # ------------------------------------------------------------------ #
        # TF2
        # ------------------------------------------------------------------ #

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        # STATE
        # ------------------------------------------------------------------ #

        self.joint_pos   = [0.0] * self.act_dim
        self.joint_vel   = [0.0] * self.act_dim
        self.object_xyz  = [0.0, 0.0, 0.0]
        self.object_received = False
        self.prev_actions = [0.0] * self.act_dim
        self.timestep = 0

        # ------------------------------------------------------------------ #
        # EE LOGGING
        # ------------------------------------------------------------------ #

        self.ee_frame = "follower/gripper_frame_link"
        self._ee_log  = []
        self._t0      = time.monotonic()

        # ------------------------------------------------------------------ #
        # ROS TOPICS
        # ------------------------------------------------------------------ #

        self.create_subscription(
            JointState, "/follower/joint_states",
            self.joint_state_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/object_pose",
            self.pose_callback, 10
        )

        self.pub = self.create_publisher(
            Float64MultiArray,
            "/follower/forward_controller/commands", 10
        )

        self.timer = self.create_timer(
            1.0 / self.control_hz,
            self.timer_callback
        )

        self.get_logger().info(
            f"RLControlNode ready | "
            f"obs_dim={self.obs_dim} act_dim={self.act_dim} freq={self.control_hz}Hz"
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def joint_state_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        missing = [n for n in self.joint_names if n not in name_to_index]

        if missing:
            self.get_logger().warn(
                f"Missing joints: {missing}", throttle_duration_sec=2.0
            )
            return

        self.joint_pos = [
            float(msg.position[name_to_index[n]]) for n in self.joint_names
        ]
        self.joint_vel = (
            [float(msg.velocity[name_to_index[n]]) for n in self.joint_names]
            if len(msg.velocity) >= len(msg.name)
            else [0.0] * len(self.joint_names)
        )

    def pose_callback(self, msg: PoseStamped):
        point_in_cam = PointStamped()
        point_in_cam.header = msg.header
        point_in_cam.header.frame_id = self.camera_frame
        point_in_cam.point.x = msg.pose.position.x
        point_in_cam.point.y = msg.pose.position.y
        point_in_cam.point.z = msg.pose.position.z

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.base_frame,
                source_frame=self.camera_frame,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            pt = tf2_geometry_msgs.do_transform_point(point_in_cam, transform)
            self.object_xyz = [pt.point.x, pt.point.y, pt.point.z]
            self.object_received = True

        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.get_logger().warn(f"TF error: {e}", throttle_duration_sec=2.0)

    # =========================================================================
    # CONTROL LOOP
    # =========================================================================

    def timer_callback(self):
        if not self.object_received:
            self.get_logger().warn(
                "Waiting for object pose...", throttle_duration_sec=2.0
            )
            return

        # --- Build observation -----------------------------------------------
        joint_pos_np = np.array(self.joint_pos, dtype=np.float32)
        joint_vel_np = np.array(self.joint_vel, dtype=np.float32)

        joint_pos_rel = (
            (joint_pos_np - self.default_joint_pos) / (self.joint_range + 1e-8)
            if self.normalize_joint_pos
            else joint_pos_np - self.default_joint_pos
        )

        pose_cmd = list(self.pose_command)

        obs = (
            joint_pos_rel.tolist()
            + joint_vel_np.tolist()
            + pose_cmd
            + self.prev_actions
        )

        if self.timestep < 3:
            self.get_logger().info(f"OBS: {obs}")


        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.use_scaler:
            obs_tensor = self.obs_scaler(obs_tensor)

        # --- Policy inference ------------------------------------------------
        with torch.inference_mode():
            outputs = self.policy.act(
                {"states": obs_tensor, "observations": obs_tensor},
                role="policy",
            )
            actions    = outputs[-1].get("mean_actions", outputs[0])
            mean_action = actions.squeeze(0).detach().cpu().numpy()

        if not np.all(np.isfinite(mean_action)):
            self.get_logger().error(f"Non-finite action: {mean_action}")
            return

        # --- Clip, scale, rate-limit, publish --------------------------------
        safe_action  = np.clip(mean_action, -self.max_abs_action, self.max_abs_action)
        joint_target = safe_action * self.action_scale + self.default_joint_pos

        if self.max_delta > 0.0:
            current      = np.array(self.joint_pos, dtype=np.float32)
            joint_target = current + np.clip(joint_target - current, -self.max_delta, self.max_delta)

        self._publish(joint_target.tolist())

        # Isaac Lab stores raw network action as prev_actions
        self.prev_actions = mean_action.tolist()
        self.timestep += 1

        # --- Log EE trajectory -----------------------------------------------
        ee_xyz = self._lookup_ee_pose()
        if ee_xyz is not None:
            desired = self.pose_command[:3]
            self._ee_log.append((
                round(time.monotonic() - self._t0, 4),
                round(ee_xyz[0], 6), round(ee_xyz[1], 6), round(ee_xyz[2], 6),
                round(desired[0], 6), round(desired[1], 6), round(desired[2], 6),
            ))

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _publish(self, commands):
        target = np.clip(np.array(commands, dtype=np.float32), self.joint_min, self.joint_max)
        msg = Float64MultiArray()
        msg.data = target.tolist()
        self.pub.publish(msg)
        return target

    def _lookup_ee_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.base_frame,
                source_frame=self.ee_frame,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0),
            )
            t = transform.transform.translation
            return (t.x, t.y, t.z)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            return None

    def _flush_ee_log(self):
        if not self._ee_log:
            self.get_logger().info("No EE trajectory data to write")
            return
        try:
            with open(self.ee_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_s", "ee_x", "ee_y", "ee_z", "desired_x", "desired_y", "desired_z"])
                writer.writerows(self._ee_log)
            self.get_logger().info(f"Saved EE trajectory: {self.ee_log_path}")
        except OSError as e:
            self.get_logger().error(f"Failed to write EE log: {e}")

    def destroy_node(self):
        self._flush_ee_log()
        super().destroy_node()


# =============================================================================
# MAIN
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = RLControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()