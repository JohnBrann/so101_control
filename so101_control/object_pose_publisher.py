import json
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import pupil_apriltags as apriltag

class AprilTagPosePublisher(Node):
    def __init__(self):
        super().__init__('apriltag_pose_publisher')

        # --- Load config ---
        self.declare_parameter('config_path', '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value

        if not config_path:
            pkg_share = get_package_share_directory('so101_control')
            config_path = os.path.join(pkg_share, 'config', 'camera_config.json')

        self.get_logger().info(f'Loading config from: {config_path}')
        with open(config_path, 'r') as f:
            config = json.load(f)

        cam_cfg = config['camera']
        tag_cfg = config['april_tag']

        # Camera intrinsics (fx, fy, cx, cy)
        K = cam_cfg['cam_K']
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.cx = K[0][2]
        self.cy = K[1][2]
        self.dist_coeffs = np.array(cam_cfg['distortion_coefficients'], dtype=np.float32)

        self.tag_size = tag_cfg['APRILTAG_SIZE']  # meters
        self.width    = cam_cfg['width']
        self.height   = cam_cfg['height']

        # --- AprilTag detector ---
        self.detector = apriltag.Detector(
            families=tag_cfg['APRILTAG_FAMILY'],
            nthreads=2,
        )

        # --- Webcam ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open webcam')
            raise RuntimeError('Webcam not available')

        # --- Publisher ---
        self.publisher_ = self.create_publisher(PoseStamped, '/object_pose', 10)
        # self.publisher_ = self.create_publisher(PoseStamped, '/follower/image_raw/compressed', 10)


        # --- Timer (20 Hz) ---
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Cache last known pose so we keep publishing even between detections
        self.last_pose: PoseStamped | None = None

        self.get_logger().info(
            f'AprilTagPosePublisher ready | tag_size={self.tag_size}m '
            f'family={tag_cfg["APRILTAG_FAMILY"]}'
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=self.tag_size,
        )

        if not detections:
            self.get_logger().debug('No AprilTag detected — republishing last known pose')
            if self.last_pose is not None:
                self.publisher_.publish(self.last_pose)

        # Use the detection with the highest decision margin if multiple found
        else:
            best = max(detections, key=lambda d: d.decision_margin)
            pose_msg = self._build_pose_msg(best)
            self.last_pose = pose_msg
            self.publisher_.publish(pose_msg)
            self.get_logger().info(
                f'Tag {best.tag_id} | '
                f'x={pose_msg.pose.position.x:.3f} '
                f'y={pose_msg.pose.position.y:.3f} '
                f'z={pose_msg.pose.position.z:.3f}'
            )

        # --- Draw detections overlay ---
        for det in detections:
            # Draw the tag border
            corners = det.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

            # Draw center dot
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw tag ID and Z distance
            x = float(self.last_pose.pose.position.x) if self.last_pose else 0.0
            y = float(self.last_pose.pose.position.y) if self.last_pose else 0.0
            z = float(self.last_pose.pose.position.z) if self.last_pose else 0.0
            # cv2.putText(frame, f'ID:{det.tag_id}  [{x:.2f},{y:.2f},{z:.2f}]',
            cv2.putText(frame, f'[{x:.2f},{y:.2f},{z:.2f}]',
                        (cx - 80, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show status text when no tag is visible
        if not detections:
            cv2.putText(frame, 'No tag detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('AprilTag Pose Estimation', frame)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    def _build_pose_msg(self, detection) -> PoseStamped:
        """Convert a pupil_apriltags Detection into a PoseStamped message."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        # Translation (meters)
        t = detection.pose_t          # shape (3, 1)
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])

        # Rotation: rotation matrix → quaternion
        R = detection.pose_R          # shape (3, 3)
        q = self._rot_to_quat(R)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        return msg

    @staticmethod
    def _rot_to_quat(R: np.ndarray):
        """Shepperd's method: rotation matrix → quaternion [x, y, z, w]."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    # ------------------------------------------------------------------
    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()