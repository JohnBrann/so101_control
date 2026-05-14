import yaml
import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory


class RandomPosePublisher(Node):
    def __init__(self):
        super().__init__('random_pose_publisher')

        # --- Load config ---
        self.declare_parameter('config_path', '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value

        if not config_path:
            pkg_share = get_package_share_directory('so101_control')
            config_path = os.path.join(pkg_share, 'config', 'random_pose.yaml')

        self.get_logger().info(f'Loading config from: {config_path}')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        params = config['random_pose_node']['ros__parameters']

        self.random      = params['random_pose']
        self.set_pose    = params['set_pose']                      # [x, y, z]
        self.x_bounds    = params['workspace']['x_bounds']         # [min, max]
        self.y_bounds    = params['workspace']['y_bounds']
        self.z_bounds    = params['workspace']['z_bounds']

        # --- Publisher ---
        self.publisher_ = self.create_publisher(PoseStamped, '/pose', 10)

        # --- Timer (10 Hz) ---
        self.timer = self.create_timer(1, self.timer_callback)

    # ------------------------------------------------------------------
    def timer_callback(self):
        if self.random:
            x = round(np.random.uniform(*self.x_bounds), 3)
            y = round(np.random.uniform(*self.y_bounds), 3)
            z = round(np.random.uniform(*self.z_bounds), 3)
        else:
            x, y, z = [round(v, 3) for v in self.set_pose]

        pose_msg = self._build_pose_msg(x, y, z)
        self.publisher_.publish(pose_msg)
        # self.get_logger().info(
        #     f'Publishing | '
        #     f'x={pose_msg.pose.position.x:.3f} '
        #     f'y={pose_msg.pose.position.y:.3f} '
        #     f'z={pose_msg.pose.position.z:.3f}'
        # )

    # ------------------------------------------------------------------
    def _build_pose_msg(self, x: float, y: float, z: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base'

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # Identity quaternion (no rotation)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 1.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0

        return msg

    # ------------------------------------------------------------------
    def destroy_node(self):
        super().destroy_node()


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RandomPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()