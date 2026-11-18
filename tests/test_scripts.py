"""Small dependency-free checks; hardware integration remains a board check."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import camera


class ScriptsTest(unittest.TestCase):
    def test_bad_arguments_fail_before_dependencies(self):
        for script, args in [
            ('collect.py', ['--topic', '/camera', '--fps', '0']),
            ('collect.py', ['--topic', '/camera', '--count', '-1']),
            ('detect.py', ['--topic', '/camera', '--model', '/missing/model.pt']),
        ]:
            with self.subTest(script=script, args=args):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_onnx_export_contract(self):
        spec = importlib.util.spec_from_file_location('export_model', ROOT / 'scripts/export.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yolo = Mock()
        with patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=yolo)}), \
                patch.object(sys, 'argv', ['export.py', __file__]):
            module.main()
        yolo.return_value.export.assert_called_once_with(
            format='onnx', imgsz=640, batch=1, device='cpu', opset=19,
            dynamic=False, simplify=True, nms=False)

    def test_latest_frame_and_header(self):
        nodes = []

        class Node:
            def __init__(self, name):
                nodes.append(self)

            def create_subscription(self, msg_type, topic, callback, qos):
                self.callback = callback

            def create_timer(self, period, callback):
                self.tick = callback

            def get_logger(self):
                return Mock()

            def destroy_node(self):
                self.destroyed = True

        bridge = Mock()
        bridge.imgmsg_to_cv2.side_effect = lambda msg, encoding: msg.data
        process = Mock()
        first = types.SimpleNamespace(data='old', header='old_header')
        last = types.SimpleNamespace(data='latest', header='camera_header')

        def spin(node):
            node.callback(first)
            node.callback(last)
            node.tick()
            node.tick()

        modules = {
            'rclpy': types.SimpleNamespace(init=Mock(), spin=spin, ok=lambda: True, shutdown=Mock()),
            'rclpy.node': types.SimpleNamespace(Node=Node),
            'rclpy.qos': types.SimpleNamespace(qos_profile_sensor_data=object()),
            'sensor_msgs': types.ModuleType('sensor_msgs'),
            'sensor_msgs.msg': types.SimpleNamespace(Image=object, CompressedImage=object),
            'cv_bridge': types.SimpleNamespace(CvBridge=lambda: bridge),
        }
        args = types.SimpleNamespace(command='detect', compressed=False, fps=5, topic='/camera')
        with patch.dict(sys.modules, modules):
            camera.run(args, process)
        process.assert_called_once_with(nodes[0], 'latest', 'camera_header')
        self.assertTrue(nodes[0].destroyed)

    def test_detector_publishes_both_images(self):
        import detect
        yolo = Mock()
        result = Mock()
        result.plot.return_value = 'annotated pixels'
        yolo.return_value.predict.return_value = [result]
        node = Mock()
        del node.image_pub
        raw, jpeg = Mock(), Mock()
        node.create_publisher.side_effect = [raw, jpeg]
        image = types.SimpleNamespace(header=None)
        compressed = types.SimpleNamespace(header=None)
        node.bridge.cv2_to_imgmsg.return_value = image
        node.bridge.cv2_to_compressed_imgmsg.return_value = compressed

        def run(args, callback):
            callback(node, 'camera pixels', 'source timestamp and frame')

        with patch.dict(sys.modules, {
                'ultralytics': types.SimpleNamespace(YOLO=yolo),
                'sensor_msgs': types.ModuleType('sensor_msgs'),
                'sensor_msgs.msg': types.SimpleNamespace(Image=object, CompressedImage=object),
        }), patch.object(detect, 'run', run), patch.object(sys, 'argv', [
                'detect.py', '--topic', '/camera', '--model', __file__]):
            detect.main()
        raw.publish.assert_called_once_with(image)
        jpeg.publish.assert_called_once_with(compressed)
        self.assertEqual(image.header, 'source timestamp and frame')
        self.assertEqual(compressed.header, image.header)
        node.bridge.cv2_to_imgmsg.assert_called_once_with('annotated pixels', encoding='bgr8')


if __name__ == '__main__':
    unittest.main()
