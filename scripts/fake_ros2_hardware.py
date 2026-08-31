"""
Fake hardware driver for dry-testing eval_crazieflieros2.py without a real
drone/camera. Publishes synthetic PoseStamped (stand-in for mavros) and
Image frames (stand-in for the realsense-ros driver) on the same topics/
encodings/QoS that eval_crazieflieros2.py subscribes to, so the whole
model-load -> sample -> project -> publish loop can be exercised end to end.

Usage (two terminals, same conda env / ROS2 setup in both):
    python scripts/fake_ros2_hardware.py --use_depth
    python scripts/eval_crazieflieros2.py --run_dir <path> --variant diffuser
        (omit --live -- dry run just prints the commands it would send)
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

FPV_WIDTH, FPV_HEIGHT = 96, 96


class FakeHardware(Node):
    def __init__(self, args):
        super().__init__("fake_hardware")
        self.args = args
        self.pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)  # start airborne, mid-corridor height

        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, qos_profile_sensor_data)
        self.color_pub = self.create_publisher(Image, args.color_topic, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, args.depth_topic, qos_profile_sensor_data) if args.use_depth else None

        self.create_timer(1.0 / args.pose_rate, self._publish_pose)
        self.create_timer(1.0 / args.image_rate, self._publish_images)

    def _publish_pose(self):
        # slow drift so the policy sees a moving drone, not a frozen one
        self.pos[0] += 0.002
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.pos.tolist()
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

    def _publish_images(self):
        stamp = self.get_clock().now().to_msg()

        color = np.random.randint(0, 255, (FPV_HEIGHT, FPV_WIDTH, 3), dtype=np.uint8)
        cmsg = Image()
        cmsg.header.stamp = stamp
        cmsg.height, cmsg.width = FPV_HEIGHT, FPV_WIDTH
        cmsg.encoding = "rgb8"
        cmsg.step = FPV_WIDTH * 3
        cmsg.data = color.tobytes()
        self.color_pub.publish(cmsg)

        if self.depth_pub is not None:
            depth = np.random.uniform(0.5, 3.0, (FPV_HEIGHT, FPV_WIDTH)).astype(np.float32)
            dmsg = Image()
            dmsg.header.stamp = stamp
            dmsg.height, dmsg.width = FPV_HEIGHT, FPV_WIDTH
            dmsg.encoding = "32FC1"
            dmsg.step = FPV_WIDTH * 4
            dmsg.data = depth.tobytes()
            self.depth_pub.publish(dmsg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_topic", type=str, default="/mavros/local_position/pose")
    parser.add_argument("--color_topic", type=str, default="camera/camera/color/image_raw")
    parser.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--use_depth", action="store_true", default=False,
                        help="Also publish fake depth frames (pass this if your checkpoint has use_depth=True).")
    parser.add_argument("--pose_rate", type=float, default=30.0)
    parser.add_argument("--image_rate", type=float, default=15.0)
    args = parser.parse_args()

    rclpy.init()
    node = FakeHardware(args)
    print(f"[fake_ros2_hardware] publishing pose on {args.pose_topic} @ {args.pose_rate}Hz, "
          f"color on {args.color_topic} @ {args.image_rate}Hz"
          + (f", depth on {args.depth_topic}" if args.use_depth else "") + ". Ctrl+C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
