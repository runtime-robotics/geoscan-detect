"""Save camera frames while the pilot flies manually."""
from pathlib import Path
from datetime import datetime, timezone
import time
from camera import arguments, validate, run


def main():
    parser = arguments(__doc__)
    parser.set_defaults(command='collect', fps=1)
    parser.add_argument('--output', type=Path, default=Path('captures'))
    parser.add_argument('--count', type=int, default=300)
    args = parser.parse_args()
    validate(parser, args)
    if args.count <= 0:
        parser.error('--count must be positive')
    import cv2
    import rclpy

    folder = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    count = 0

    def save(node, frame, header):
        nonlocal count
        path = folder / f'{count:05d}_{time.time_ns()}.jpg'
        if not cv2.imwrite(str(path), frame):
            raise OSError('Cannot write ' + str(path))
        count += 1
        node.get_logger().info(f'{count}/{args.count}: {path}')
        if count >= args.count:
            rclpy.shutdown()

    run(args, save)


if __name__ == '__main__':
    main()
