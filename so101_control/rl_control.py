import os
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import tf2_ros
import tf2_geometry_msgs


# ---------------------------------------------------------------------------
# Network — mirrors SharedModel from training output exactly:
#
#   net_container: LazyLinear(128) -> ELU -> LazyLinear(128) -> ELU  (shared trunk)
#   policy_layer:  LazyLinear(act_dim)                               (policy head)
#   log_std_parameter: Parameter(act_dim)                            (not used at inference)
#
# GaussianMixin with clip_actions=True applies tanh to policy_layer output
# before returning mean_actions. We replicate that here.
#
# Checkpoint key structure (from skrl):
#   policy/net_container.0.weight
#   policy/net_container.0.bias
#   policy/net_container.2.weight
#   policy/net_container.2.bias
#   policy/policy_layer.weight
#   policy/policy_layer.bias
#   policy/log_std_parameter
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 128),  # matches LazyLinear(128) after initialization
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(128, act_dim)
        self.log_std_parameter = nn.Parameter(
            torch.full((act_dim,), 0.0), requires_grad=False  # not needed at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors: compute() returns raw output, then GaussianMixin.act() applies tanh
        # because clip_actions=True
        net = self.net_container(x)
        raw = self.policy_layer(net)
        return torch.tanh(raw)  # mean_actions in [-1, 1]


# ---------------------------------------------------------------------------
# Running Standard Scaler
# ---------------------------------------------------------------------------
class RunningStandardScaler:
    def __init__(self, device):
        self.device = device
        self.running_mean     = None
        self.running_variance = None
        self.count            = None

    def load(self, state_dict: dict, prefix: str):
        self.running_mean     = state_dict[f'{prefix}running_mean'].to(self.device)
        self.running_variance = state_dict[f'{prefix}running_variance'].to(self.device)
        self.count            = state_dict[f'{prefix}count'].to(self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.running_variance / self.count.clamp(min=1))
        return (x - self.running_mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Action space configuration — matches Isaac Lab task config:
#
#   JointPositionActionCfg(scale=0.5, use_default_offset=True)
#
#   final_joint_pos = mean_action * ACTION_SCALE + DEFAULT_JOINT_POS
#
# DEFAULT_JOINT_POS: fill in from SO_ARM101_CFG init_state.joint_pos
# ---------------------------------------------------------------------------
ACTION_SCALE = 0.5

DEFAULT_JOINT_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # <-- UPDATE FROM SO_ARM101_CFG

# Hardcoded gripper orientation command — must match sim ee_pose command format
# Isaac Lab UniformPoseCommandCfg uses [x, y, z, qw, qx, qy, qz] internally
# Verify this order matches your obs space by comparing to your frozen sim obs block
GRIPPER_DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]  # [qw, qx, qy, qz] for pitch=pi

# Home pose — raw joint positions in radians sent directly to controller
HOME_POSE = [-0.02, -1.81, 1.41, 0.98, 0.08, -0.61]

EPISODE_DURATION   = 5.0   # seconds before homing
HOME_HOLD_DURATION = 5.0   # seconds held at home before resuming


# ---------------------------------------------------------------------------
class RLControlNode(Node):
    def __init__(self):
        super().__init__('rl_control_node')

        # --- Parameters ---
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('obs_dim', 25)
        self.declare_parameter('act_dim', 6)

        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        obs_dim         = self.get_parameter('obs_dim').get_parameter_value().integer_value
        act_dim         = self.get_parameter('act_dim').get_parameter_value().integer_value

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self.get_logger().error(f'Checkpoint not found: {checkpoint_path}')
            raise FileNotFoundError(checkpoint_path)

        # --- Device ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Load checkpoint ---
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.get_logger().info(f'Checkpoint keys (first 20): {list(ckpt.keys())[:20]}')

        # --- Build key mapping from checkpoint to our PolicyNetwork ---
        # Checkpoint uses: policy/net_container.X.weight
        # Our module uses: net_container.X.weight
        # Also skip value_layer keys — we don't need them
        policy_state = {}
        for k, v in ckpt.items():
            if not k.startswith('policy/'):
                continue
            new_key = k[len('policy/'):]
            # Skip value head weights — not part of PolicyNetwork
            if new_key.startswith('value_layer'):
                continue
            policy_state[new_key] = v

        self.get_logger().info(f'Policy state keys being loaded: {list(policy_state.keys())}')

        # --- Policy ---
        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)
        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        self.get_logger().info(f'Policy missing keys  : {missing}')
        self.get_logger().info(f'Policy unexpected    : {unexpected}')

        # Flag any weight mismatches (log_std missing is fine, weight mismatches are not)
        weight_missing = [k for k in missing if 'log_std' not in k]
        if weight_missing:
            self.get_logger().error(
                f'WEIGHT MISMATCH — these keys did not load: {weight_missing}. '
                f'Policy will produce garbage. Check architecture against checkpoint.'
            )
        else:
            self.get_logger().info('All weight keys loaded successfully.')

        self.policy.eval()

        # --- Observation scaler ---
        self.obs_scaler = RunningStandardScaler(self.device)
        try:
            self.obs_scaler.load(ckpt, prefix='state_preprocessor/')
            self.use_scaler = True
            self.get_logger().info('Observation scaler loaded.')
        except KeyError as e:
            self.get_logger().warn(f'Scaler key not found ({e}) — running without scaling.')
            self.use_scaler = False

        # --- TF2 ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.base_frame   = 'follower/base_link'
        self.camera_frame = 'follower/cam_wrist'

        # --- State ---
        self.joint_pos       = [0.0] * 6
        self.joint_vel       = [0.0] * 6
        self.object_xyz      = [0.0, 0.0, 0.0]
        self.object_received = False
        # Stores the previous *policy output* (tanh'd, [-1,1]) — NOT joint targets
        # This is what the sim echoes back in obs[19:25]
        self.prev_actions    = [0.0] * act_dim

        # --- Episode timing ---
        self.mode          = 'policy'
        self.episode_start = self.get_clock().now()
        self.home_start    = None

        # --- ROS topics ---
        self.create_subscription(JointState,  '/follower/joint_states', self.joint_state_callback, 10)
        self.create_subscription(PoseStamped, '/object_pose',           self.pose_callback,        10)
        self.pub   = self.create_publisher(Float64MultiArray, '/follower/forward_controller/commands', 10)
        # 50 Hz to match Isaac Lab default control frequency
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info(f'Ready | obs_dim={obs_dim} act_dim={act_dim} device={self.device}')
        self.get_logger().info(
            f'Action pipeline: tanh(policy_layer(net_container(obs))) '
            f'* {ACTION_SCALE} + {DEFAULT_JOINT_POS.tolist()}'
        )

    # -----------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        if len(msg.position) >= 6:
            self.joint_pos = list(msg.position[:6])
        if len(msg.velocity) >= 6:
            self.joint_vel = list(msg.velocity[:6])

    def pose_callback(self, msg: PoseStamped):
        point_in_cam = PointStamped()
        point_in_cam.header          = msg.header
        point_in_cam.header.frame_id = self.camera_frame
        point_in_cam.point.x         = msg.pose.position.x
        point_in_cam.point.y         = msg.pose.position.y
        point_in_cam.point.z         = msg.pose.position.z

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.base_frame,
                source_frame=self.camera_frame,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            point_in_base   = tf2_geometry_msgs.do_transform_point(point_in_cam, transform)
            self.object_xyz = [point_in_base.point.x,
                               point_in_base.point.y,
                               point_in_base.point.z]
            self.object_received = True
            self.get_logger().debug(
                f'Object in base frame: {[f"{v:.4f}" for v in self.object_xyz]}'
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f'TF error: {e}', throttle_duration_sec=2.0)

    # -----------------------------------------------------------------------
    def _publish(self, commands: list):
        out      = Float64MultiArray()
        out.data = commands
        self.pub.publish(out)

    def _elapsed(self, since) -> float:
        return (self.get_clock().now() - since).nanoseconds * 1e-9

    # -----------------------------------------------------------------------
    def timer_callback(self):

        # ---- HOMING mode ----
        if self.mode == 'homing':
            self._publish(HOME_POSE)
            if self._elapsed(self.home_start) >= HOME_HOLD_DURATION:
                self.get_logger().info('Home hold complete — resuming policy.')
                self.prev_actions  = [0.0] * 6
                self.episode_start = self.get_clock().now()
                self.mode          = 'policy'
            return

        # ---- POLICY mode ----
        if not self.object_received:
            self.get_logger().warn('Waiting for object pose...', throttle_duration_sec=2.0)
            return

        if self._elapsed(self.episode_start) >= EPISODE_DURATION:
            self.get_logger().info(
                f'Episode complete ({EPISODE_DURATION}s) — returning to home pose.'
            )
            self.mode       = 'homing'
            self.home_start = self.get_clock().now()
            self._publish(HOME_POSE)
            return

        # --- Build observation — must match sim obs space exactly ---
        # [joint_pos(6), joint_vel(6), ee_pose_command(7), prev_actions(6)] = 25
        pose_command = self.object_xyz + GRIPPER_DOWN_QUAT
        obs = (
            self.joint_pos   +   # 6  joint positions (rad)
            self.joint_vel   +   # 6  joint velocities (rad/s)
            pose_command     +   # 7  target ee pose in robot base frame
            self.prev_actions    # 6  previous policy output (tanh'd, [-1, 1])
        )

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.use_scaler:
            obs_tensor = self.obs_scaler(obs_tensor)

        with torch.inference_mode():
            # forward() = tanh(policy_layer(net_container(obs)))
            # output is mean_actions in [-1, 1], matching skrl clip_actions=True
            mean_action = self.policy(obs_tensor).squeeze(0).cpu().numpy()

        # --- Replicate JointPositionActionCfg(scale=0.5, use_default_offset=True) ---
        joint_target = mean_action * ACTION_SCALE + DEFAULT_JOINT_POS

        self.get_logger().info(
            f'[{self._elapsed(self.episode_start):.1f}s/{EPISODE_DURATION}s] '
            f'mean_action={[f"{a:.3f}" for a in mean_action.tolist()]} | '
            f'joint_target={[f"{j:.3f}" for j in joint_target.tolist()]}'
        )

        # Store policy output (NOT joint target) for next obs
        self.prev_actions = mean_action.tolist()

        self._publish(joint_target.tolist())

    # -----------------------------------------------------------------------
    def destroy_node(self):
        super().destroy_node()


# ---------------------------------------------------------------------------
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

if __name__ == '__main__':
    main()