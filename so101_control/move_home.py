import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

TARGET_POSE = [0.0, 0.0, 0.0, 1.57, 0.0, 0.0]
# Order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

MAX_DELTA_PER_STEP = 0.02  # rad per tick — slow and safe
PUBLISH_HZ = 30.0

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

JOINT_MIN = np.array([-1.7279, -1.5708, -1.5210, -1.4923, -2.4695, -0.1571])
JOINT_MAX = np.array([ 1.7279,  1.5708,  1.5210,  1.4923,  2.5571,  1.5708])


class GoToDefaultNode(Node):
    def __init__(self):
        super().__init__('go_to_default_node')

        self.joint_pos = None
        self.last_command = None
        self.at_target = False

        self.create_subscription(
            JointState,
            '/follower/joint_states',
            self.joint_state_callback,
            10,
        )
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/follower/forward_controller/commands',
            10,
        )
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)
        self.get_logger().info(
            f'Moving to default pose: {TARGET_POSE} at {PUBLISH_HZ} Hz, '
            f'max delta {MAX_DELTA_PER_STEP} rad/tick'
        )

    def joint_state_callback(self, msg: JointState):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        missing = [n for n in JOINT_ORDER if n not in name_to_index]
        if missing:
            self.get_logger().warn(
                f'Missing joints: {missing}', throttle_duration_sec=2.0
            )
            return
        self.joint_pos = np.array([
            msg.position[name_to_index[n]] for n in JOINT_ORDER
        ])

    def timer_callback(self):
        if self.joint_pos is None:
            self.get_logger().warn(
                'Waiting for joint states...', throttle_duration_sec=2.0
            )
            return

        if self.at_target:
            return

        # Seed last_command from current position on first publish
        if self.last_command is None:
            self.last_command = self.joint_pos.copy()

        target = np.clip(np.array(TARGET_POSE), JOINT_MIN, JOINT_MAX)

        delta = target - self.last_command
        delta_limited = np.clip(delta, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
        command = self.last_command + delta_limited
        self.last_command = command.copy()

        out = Float64MultiArray()
        out.data = command.tolist()
        self.pub.publish(out)

        error = np.max(np.abs(target - self.joint_pos))
        self.get_logger().info(
            f'cmd={[f"{v:.3f}" for v in command]} | '
            f'pos={[f"{v:.3f}" for v in self.joint_pos]} | '
            f'max_err={error:.4f} rad',
            throttle_duration_sec=0.5,
        )

        if error < 0.02:
            self.get_logger().info('Reached default pose. Holding.')
            self.at_target = True


def main(args=None):
    rclpy.init(args=args)
    node = GoToDefaultNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()