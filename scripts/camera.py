"""Shared latest-frame ROS subscription. Does not open a camera device."""
import time


def run(args, process):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CompressedImage
    from cv_bridge import CvBridge

    class CameraNode(Node):
        def __init__(self):
            super().__init__('geoscan_' + args.command)
            self.latest = None
            self.bridge = CvBridge()
            self.sub = self.create_subscription(
                CompressedImage if args.compressed else Image,
                args.topic, self.receive, qos_profile_sensor_data)
            self.timer = self.create_timer(1 / args.fps, self.tick)
            self.started = time.monotonic()
            self.seen = False
            self.get_logger().info('Waiting for ' + args.topic)

        def receive(self, msg):
            self.latest = msg
            self.seen = True

        def tick(self):
            if self.latest is None:
                if not self.seen and time.monotonic() - self.started > 15:
                    raise RuntimeError('No camera frames for 15 s; check topic, type and ROS_DOMAIN_ID')
                return
            msg, self.latest = self.latest, None
            frame = (self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8') if args.compressed
                     else self.bridge.imgmsg_to_cv2(msg, 'bgr8'))
            process(self, frame, msg.header)

    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def arguments(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--topic', required=True, help='Existing Geoscan camera topic')
    parser.add_argument('--compressed', action='store_true', help='Input is sensor_msgs/CompressedImage')
    parser.add_argument('--fps', type=float, default=5)
    return parser


def validate(parser, args):
    import math
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be finite and positive')
