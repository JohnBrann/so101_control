#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class SimpleMove(Node):
    def __init__(self):
        super().__init__("simple_move")

        self.pub = self.create_publisher(
            Float64MultiArray,
            "/follower/forward_controller/commands",
            10,
        )

        self.start_pose = [-0.02, -1.81, 1.41, 0.98, 0.08, -0.61]
        self.end_pose   = [-0.18, -0.58, 0.27, 1.14, -0.12, -0.61]

        self.rate_hz = 30.0
        self.num_steps = 10
        self.hold_start_steps = 50
        self.hold_end_steps = 50

        self.run_motion()

    def publish_pose(self, pose):
        msg = Float64MultiArray()
        msg.data = pose
        self.pub.publish(msg)


    # Given start and end joint poses, make a trajectory to get between these points given total number of steps. 
    def make_trajectory(self, start_pose, end_pose, num_steps):
        trajectory = []

        for i in range(num_steps + 1):
            alpha = i / num_steps
            pose = [
                (1 - alpha) * start + alpha * end
                for start, end in zip(start_pose, end_pose)
            ]
            trajectory.append(pose)

        return trajectory

    def run_motion(self):
        sleep_dt = 1.0 / self.rate_hz

        self.get_logger().info("Holding start pose...")
        for _ in range(self.hold_start_steps):
            self.publish_pose(self.start_pose)
            time.sleep(sleep_dt)

        self.get_logger().info("Generating trajectory...")
        trajectory = self.make_trajectory(
            self.start_pose,
            self.end_pose,
            self.num_steps,
        )

        self.get_logger().info("Publishing trajectory...")
        for pose in trajectory:
            self.publish_pose(pose)
            time.sleep(sleep_dt)

        self.get_logger().info("Holding end pose...")
        for _ in range(self.hold_end_steps):
            self.publish_pose(self.end_pose)
            time.sleep(sleep_dt)

        self.get_logger().info("Returning to starting pose...")
        for pose in reversed(trajectory):
            self.publish_pose(pose)
            time.sleep(sleep_dt)

        self.get_logger().info("Done.")


def main():
    rclpy.init()
    node = SimpleMove()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()