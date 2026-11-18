"""Run YOLO11 and publish annotated ROS images, without flight commands."""
from pathlib import Path
from camera import arguments, validate, run


def main():
    parser = arguments(__doc__)
    parser.set_defaults(command='detect')
    parser.add_argument('--model', type=Path, required=True, help='.pt or exported *_rknn_model directory')
    parser.add_argument('--output-topic', default='/geoscan_detect/image')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.4)
    args = parser.parse_args()
    validate(parser, args)
    if not args.model.exists():
        parser.error('Model path does not exist')
    if args.topic in (args.output_topic, args.output_topic + '/compressed'):
        parser.error('Input and output topics must differ')
    if not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    from ultralytics import YOLO
    from sensor_msgs.msg import Image, CompressedImage

    model = YOLO(str(args.model.resolve()), task='detect')

    def detect(node, frame, header):
        if not hasattr(node, 'image_pub'):
            node.image_pub = node.create_publisher(Image, args.output_topic, 1)
            node.jpeg_pub = node.create_publisher(CompressedImage, args.output_topic + '/compressed', 1)
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device='cpu', verbose=False)[0]
        annotated = result.plot()
        image = node.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        image.header = header
        node.image_pub.publish(image)
        jpeg = node.bridge.cv2_to_compressed_imgmsg(annotated, dst_format='jpg')
        jpeg.header = header
        node.jpeg_pub.publish(jpeg)

    run(args, detect)


if __name__ == '__main__':
    main()
