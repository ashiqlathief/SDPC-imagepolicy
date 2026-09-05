"""
Fake hardware rig for testing eval_crazieflieros2.py without a real drone/camera.

Publishes a synthetic /mavros/local_position/pose that slowly integrates toward
whatever eval_crazieflieros2.py publishes on /mpc/set_pose (simulating a slow
physical flight instead of teleporting), plus a constant dummy RGB frame on
camera/camera/color/image_raw. Lets you watch the arrival-gating and
stop-at-goal logic fire against real ROS2 message traffic.

Usage (two terminals):
    Terminal 1: python scripts/fake_ros2_rig.py
    Terminal 2: python scripts/eval_crazieflieros2.py   (after setting RUN_DIR/VARIANT)

Only covers the pose+color topics (VARIANT="diffuser", use_depth=False). Add a
depth + depth/camera_info publisher below if you need to test an "sdpc-*" variant.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

START_POS = np.array([3.0, 0.0, 1.0], dtype=np.float32)  # ~1m from the default target -- should
                                                           # take a few replan+hold cycles to arrive
MAX_STEP_PER_TICK = 0.03   # m -- how far the "drone" moves toward the commanded setpoint per tick
TICK_HZ = 20.0
IMG_SIZE = 96              # matches FPV_WIDTH/FPV_HEIGHT in depth_obstacle_estimator.py


class FakeRig(Node):
    def __init__(self):
        super().__init__("fake_ros2_rig")
        self.pos = START_POS.copy()
        self.cmd = None

        self.pose_pub = self.create_publisher(PoseStamped, "/mavros/local_position/pose", qos_profile_sensor_data)
        self.color_pub = self.create_publisher(Image, "camera/camera/color/image_raw", qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/mpc/set_pose", self._cmd_cb, 1)

        self._frame = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)  # flat grey, content doesn't matter here
        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(f"[FAKE RIG] start_pos={self.pos.tolist()} publishing at {TICK_HZ}Hz")

    def _cmd_cb(self, msg):
        p = msg.pose.position
        self.cmd = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.get_logger().info(f"[FAKE RIG] received setpoint: {self.cmd.tolist()}")

    def _tick(self):
        if self.cmd is not None:
            delta = self.cmd - self.pos
            dist = float(np.linalg.norm(delta))
            step = min(MAX_STEP_PER_TICK, dist)
            if dist > 1e-6:
                self.pos += (delta / dist) * step

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in self.pos)
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        img = Image()
        img.header.stamp = pose.header.stamp
        img.height, img.width = IMG_SIZE, IMG_SIZE
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = IMG_SIZE * 3
        img.data = self._frame.tobytes()
        self.color_pub.publish(img)


def main():
    rclpy.init()
    node = FakeRig()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
